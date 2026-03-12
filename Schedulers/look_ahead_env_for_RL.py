import numpy as np
import gymnasium as gym
from gymnasium import spaces

class LookAheadRLEnergyEnvSwitch(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, elec_demand_series, heat_demand_series, pv_series, components,
                 gas_price=0.11, grid_price=0.3, feed_in_price=0.065, penalty_cost=5,
                 chp_switch_cost=10, boiler_switch_cost=10, lookahead=24):
        super().__init__()

        # Data
        self.elec_demand = elec_demand_series.values.astype(float)
        self.heat_demand = heat_demand_series.values.astype(float)
        self.pv_gen = pv_series.values.astype(float)
        self.n_hours = len(self.elec_demand)
        self.lookahead = lookahead
        self.episode_t = 0      # step index within 24h game
        self.start_hour = 0     # absolute hour pointer

        # Components
        self.chp = components["chp"]
        self.boiler = components["boiler"]
        self.ee = components["ee"]
        self.pv = components["pv"]
        self.batt = components["battery"]
        self.buffer = components["buffer"]

        # Prices
        self.gas_price = gas_price
        self.grid_price = grid_price
        self.feed_in_price = feed_in_price
        self.penalty_cost = penalty_cost
        self.chp_switch_cost = float(chp_switch_cost)
        self.boiler_switch_cost = float(boiler_switch_cost)

        # SOC initialization
        self.soc_batt = float(30)
        self.soc_buff = float(50)

        # Hour pointer
        self.hour = 0
        self.episode_start = 0
        self._has_reset = False

        # For switch-cost tracking (previous on/off states)
        self.prev_chp_on = 0.0  # 0=off, 1=on
        self.prev_boiler_on = 0.0

        # [chp_power, batt_discharge, buff_discharge, boiler_power, boiler_to_buffer]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )

        # Observation: [elec_demand, heat_demand, PV_gen, SOC_batt, SOC_buff]
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(3*self.lookahead + 3,),  # 3 signals * lookahead + 2 SOCs + time-in-episode
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Advance to next 24h game
        # Note: keep SOC persistent across episodes (as you intended).
        if self._has_reset:
            self.start_hour += self.lookahead
        self._has_reset = True

        if self.start_hour + self.lookahead >= self.n_hours:
            self.start_hour = 0  # or end training

        self.episode_t = 0
        self.hour = self.start_hour  # keep a convenient "absolute hour" pointer

        # Reset switch tracking at the start of each episode
        self.prev_chp_on = 0.0
        self.prev_boiler_on = 0.0

        # Reset only episodic accumulators
        self.total_gas_cost = 0.0
        self.total_elec_cost = 0.0
        self.total_unmet_heat = 0.0
        self.total_waste_heat = 0.0

        return self._get_obs(), {}

    def _get_obs(self):
        # Fixed 24h "game window" per episode (day-ahead frame).
        # Episode 1 observes [start_hour .. start_hour+23] for all its steps.
        block_start = self.start_hour
        block_end = min(self.start_hour + self.lookahead, self.n_hours)

        elec_obs = self.elec_demand[block_start:block_end]
        heat_obs = self.heat_demand[block_start:block_end]
        pv_obs = self.pv_gen[block_start:block_end]

        # Pad if block < lookahead (only at dataset end)
        pad_len = self.lookahead - len(elec_obs)
        if pad_len > 0:
            elec_obs = np.pad(elec_obs, (0, pad_len), 'constant')
            heat_obs = np.pad(heat_obs, (0, pad_len), 'constant')
            pv_obs = np.pad(pv_obs, (0, pad_len), 'constant')

        # Include time-in-episode so the policy can choose different actions per hour
        # while keeping the 24h forecast frame fixed.
        t_frac = float(self.episode_t) / float(self.lookahead) if self.lookahead > 0 else 0.0

        obs = np.concatenate([elec_obs, heat_obs, pv_obs, [self.soc_batt, self.soc_buff, t_frac]])
        return obs.astype(np.float32)

    def step(self, action):
        # SB3 PPO can output negative actions. If the env action space is [0,1],
        # those negatives get clipped to 0 and some dims can look "stuck at zero".
        # Accept [-1,1] and map to [0,1] for internal use.
        action_raw = np.array(action, dtype=np.float32).reshape(-1)
        action_raw = np.clip(action_raw, -1.0, 1.0)
        action = 0.5 * (action_raw + 1.0)  # -> [0, 1]
        action = np.clip(action, 0.0, 1.0)

        # Absolute timestep into the dataset for this step
        t = self.start_hour + self.episode_t
        if t >= self.n_hours:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {}


        elec_req = self.elec_demand[t]
        heat_req = self.heat_demand[t]
        pv_available = max(0, self.pv_gen[t])

        # Start-of-step SOC (for feasibility layer: LP needs these to limit discharge/charge)
        soc_elec_start = float(self.soc_batt)
        soc_heat_start = float(self.soc_buff)

        # --- CHP (single continuous action) ---
        # action[0] in [0,1]: 0 = off, 1 = full.
        # Enforce a minimum load of 60% when on, and a small deadzone near 0.
        a_chp = float(action[0])
        if a_chp < 0.05:
            chp_frac = 0.0
            chp_on = 0.0
        else:
            chp_frac = 0.6 + 0.4 * a_chp  # in [0.6, 1.0]
            chp_on = 1.0

        chp_in = chp_frac * self.chp.rated_input_heat

        # --- Boiler (new continuous action) ---
        # action[3] in [0,1]: 0 = off, 1 = full. Enforce 20–100% when on.
        a_boiler = float(action[3])
        if a_boiler < 0.05:
            boiler_frac = 0.0
            boiler_on = 0.0
        else:
            boiler_frac = 0.2 + 0.8 * a_boiler  # in [0.2, 1.0]
            boiler_on = 1.0

        boiler_out_total = boiler_frac * self.boiler.rated_output_heat

        # Electricity demand met
        chp_total_elec = chp_in * (self.chp.rated_output_elec / self.chp.rated_input_heat) if self.chp.rated_input_heat > 0 else 0.0
        batt_discharge = np.clip(action[1] * min(self.batt.max_discharge_rate_elec, self.soc_batt),
                                 0, min(self.batt.max_discharge_rate_elec, self.soc_batt))
        elec_produced = chp_total_elec + pv_available + batt_discharge
        remaining_elec_demand = elec_req - elec_produced
        if remaining_elec_demand < 0:
            grid = 0.0
            excess_elec = -remaining_elec_demand
        else:
            grid = remaining_elec_demand
            excess_elec = 0.0

        # use excess for ee
        remaining_excess = excess_elec
        ee_in_actual = max(0, min(excess_elec, self.ee.rated_input_elec))
        remaining_excess -= ee_in_actual

        # store excess in battery
        self.soc_batt -= batt_discharge / self.batt.eta_discharge_elec # update SOC after discharge
        desired = remaining_excess
        input_elec = max(0, min(desired, self.batt.max_charge_rate_elec))
        to_store = input_elec * self.batt.eta_charge_elec
        available_capacity = self.batt.capacity - self.soc_batt
        actual_charge_batt = min(to_store, available_capacity)
        self.soc_batt += actual_charge_batt

        #sell remaining excess
        feed_in_value = remaining_excess - actual_charge_batt / self.batt.eta_charge_elec
        
        # --- Heat calculation ---
        buff_discharge = np.clip(action[2] * self.soc_buff * self.buffer.eta_discharge_heat,
                        0, self.soc_buff * self.buffer.eta_discharge_heat)
        
        chp_out_heat = chp_in * (self.chp.rated_output_heat / self.chp.rated_input_heat)
        ee_out_heat = ee_in_actual * (self.ee.rated_output_heat / self.ee.rated_input_elec)
        #boiler_out = boiler_in * (self.boiler.rated_output_heat / self.boiler.rated_input_heat)

        total_heat_produced = chp_out_heat + ee_out_heat + buff_discharge

        remaining_heat_demand = heat_req - total_heat_produced
        if remaining_heat_demand < 0:
            excess_heat = -remaining_heat_demand
            unmet_heat = 0.0
        else:
            excess_heat = 0.0
            unmet_heat = remaining_heat_demand
        unmet_heat_pre_boiler = float(unmet_heat)

        # Update buffer SOC after discharge (as before)
        self.soc_buff -= buff_discharge / self.buffer.eta_discharge_heat
        buffer_free_capacity = self.buffer.capacity - self.soc_buff

        # Boiler allocation: first to direct demand, then to buffer using action[4]
        boiler_out_dh = min(max(0.0, remaining_heat_demand), boiler_out_total)
        boiler_headroom = max(0.0, boiler_out_total - boiler_out_dh)

        boiler_to_buffer = min(boiler_headroom * float(action[4]), buffer_free_capacity)

        # Enforce 20% min-load rule: if total boiler output is below 20% rated, turn it off
        if boiler_out_dh + boiler_to_buffer <= 0.2 * self.boiler.rated_output_heat:
            boiler_to_buffer = 0.0
            boiler_out_dh = 0.0
            boiler_on = 0.0  # override on-flag if below min-load

        unmet_heat -= boiler_out_dh
        unmet_heat_post_boiler = float(unmet_heat)

        boiler_out = boiler_out_dh + boiler_to_buffer
        boiler_in = boiler_out * (self.boiler.rated_input_heat / self.boiler.rated_output_heat)
        
        remaining_excess_heat = excess_heat + boiler_to_buffer

        # store excess heat
        desired_heat = remaining_excess_heat
        input_heat = max(0,desired_heat)
        to_store = input_heat * self.buffer.eta_charge_heat
        available_capacity = self.buffer.capacity - self.soc_buff
        actual_charge_heat = min(to_store, available_capacity)
        self.soc_buff += actual_charge_heat

        waste_heat = remaining_excess_heat - actual_charge_heat / self.buffer.eta_charge_heat

        # Calculate costs
        gas_cost = chp_in * self.gas_price + boiler_in * self.gas_price
        elec_cost = grid * self.grid_price
        elec_total = elec_cost - feed_in_value * self.feed_in_price

        # Strong penalties for violations
        unmet_heat_penalty = 1000.0 * (unmet_heat ** 2)   # scales correctly with kWh
        waste_heat_penalty = 2.0 * waste_heat            # keep linear or also square if you want

        # Switching this step (1 if on/off changed from previous step, else 0)
        chp_switched = 1.0 if (chp_on != self.prev_chp_on) else 0.0
        boiler_switched = 1.0 if (boiler_on != self.prev_boiler_on) else 0.0
        switch_cost = (
            self.chp_switch_cost * chp_switched
            + self.boiler_switch_cost * boiler_switched
        )

        reward = -(
            gas_cost
            + elec_total
            + unmet_heat_penalty
            + waste_heat_penalty
            + switch_cost
        )

        # Update previous on/off flags for next step
        self.prev_chp_on = chp_on
        self.prev_boiler_on = boiler_on

        # Info dictionary
        info = {
        # Time alignment (use this to join demand series correctly)
        "t": int(t),
        "start_hour": int(self.start_hour),
        "episode_t": int(self.episode_t),

        # Debug: raw policy outputs vs mapped actions used by env
        "action_raw_0": float(action_raw[0]),
        "action_raw_1": float(action_raw[1]),
        "action_raw_2": float(action_raw[2]),
        "action_raw_3": float(action_raw[3]),
        "action_raw_4": float(action_raw[4]),
        "action_01_0": float(action[0]),
        "action_01_1": float(action[1]),
        "action_01_2": float(action[2]),
        "action_01_3": float(action[3]),
        "action_01_4": float(action[4]),
        "boiler_headroom": float(boiler_headroom),
        "buffer_free_capacity": float(buffer_free_capacity),
        "heat_req": float(heat_req),
        "chp_on": float(chp_on),
        "boiler_on": float(boiler_on),
        "chp_switched": float(chp_switched),
        "boiler_switched": float(boiler_switched),
        "total_heat_produced_pre_boiler": float(total_heat_produced),
        "remaining_heat_demand_pre_boiler": float(remaining_heat_demand),
        "unmet_heat_pre_boiler": float(unmet_heat_pre_boiler),
        "unmet_heat_post_boiler": float(unmet_heat_post_boiler),

        "boiler": boiler_out_dh,
        "boiler_to_buffer": boiler_to_buffer,
        "boiler_out_total": boiler_out,
        "boiler_in_total": boiler_in,
        "Grid": grid,
        "chp_out_de": chp_total_elec,
        #"chp_out_ee": chp_ee_arr,
        #"chp_out_charge": chp_batt_arr,
        "chp_out_heat": chp_out_heat,
        "chp_out_hd":chp_out_heat,
        #"chp_out_buff":chp_excess,
        "chp_in": chp_in,
        "chp_total_elec": chp_total_elec,
        "pv_out_de":pv_available,
        #"pv_out_ee":safe_arr(pv_ee,W),
        #"pv_out_charge":safe_arr(pv_batt,W),
        "buffer_out":buff_discharge,
        "total_stored_heat":desired_heat,
        "actual_stored_heat":actual_charge_heat,
        "batt_out_de":batt_discharge,
        #"batt_out_ee":batt_ee_arr,
        "total_battery_discharge":batt_discharge,
        "soc_elec": self.soc_batt,
        "actual_charged_batt": actual_charge_batt,
        "soc_heat": self.soc_buff,
        "ee_in_elec_actual":ee_in_actual,
        "ee_heat":ee_out_heat,
        "gas_price_boiler": boiler_in * self.gas_price,
        "elec_price_Grid": elec_cost,
        "gas_price_chp": chp_in * self.gas_price,
        "gas_price_total": gas_cost,
        "elec_price_total": elec_total,
        "waste_heat": waste_heat,
        "unmet_heat": unmet_heat,
        "excess_heat": excess_heat,

    }


        # Accumulate costs (within the fixed 24h episode window)
        self.total_gas_cost += gas_cost
        self.total_elec_cost += elec_total
        self.total_unmet_heat += unmet_heat
        self.total_waste_heat += waste_heat

        self.episode_t += 1
        self.hour = self.start_hour + self.episode_t

        terminated = self.episode_t >= self.lookahead
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape)

        return obs, reward, terminated, truncated, info