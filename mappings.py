# mappings.py

DEVICE_TYPE_MAPPING = {
    'loner_900': 9, 'loner_g7x': 16, 'bridge': 8, 'loner_g7l': 18, 'loner_smd': 4,
    'loner_atex': 11, 'loner_g7r': 21, 'loner_g7u': 22, 'exo': 19, 'loner_m7': 13,
    'loner_mobile': 7, 'exo_mk2': 26, 'exo8_gamma': 27, 'exo_refresh': 25, 'loner_g7p': 17,
    'loner_is': 5, 'bridge_g7': 14, 'bridge_lte': 20, 'bridge_refresh': 24, 'loner_m6i': 15
}

EVENT_TYPE_MAPPING = {
    'emergency_alert': 43, 'gas_alert_detected': 140, 'bridge_message_alert': 138,
    'over_limit_gas_alert_detected': 141, 'stel_alert_detected': 143, 'silent_alert': 42,
    'missed_check_in': 73, 'twa_alert_detected': 142, 'no_motion_occurred': 41,
    'fall_detected_alert': 40, 'logon': 28, 'server_generated_noisy_alert': 129,
    'low_battery': 14, 'network_timeout': 30, 'logoff': 29,
    'server_generated_silent_alert': 128, 'pump_block_detected': 180, 'device_tipped_over': 191
}

RESOLUTION_REASON_MAPPING = {
    'auto_resolved_by_system': 0, 'auto_resolved_by_move': 1, 'incident_without_dispatch': 2,
    'false_alert_without_dispatch': 3, 'incident_with_dispatch': 4, 'false_alert_with_dispatch': 5
}

INDUSTRY_MAPPING = {
    'oil_and_gas': 0, 'water_and_wastewater': 1, 'hazmat_and_fire_response': 2, 'utilities': 3,
    'renewable_energy': 4, 'petrochemical': 5, 'transportation_and_logistics': 6,
    'pulp_paper_and_wood_products_manufacturing': 7, 'steel_manufacturing': 8,
    'biotech_and_pharma_manufacturing': 9, 'food_processing': 10, 'education': 11,
    'agriculture': 12, 'mining': 13
}

SENSOR_TYPE_MAPPING = {
    "None": 0, 'H2S': 1, 'LEL-MPS': 2, 'O2': 3, 'NH3': 4, 'LEL': 5, 'O3': 6, 'HCN': 7,
    'Cl2': 8, 'CO': 9, 'ClO2': 10, 'PID': 11, 'HF': 12, 'NO2': 13, 'Gamma': 14, 'CO2': 15,
    'SO2': 16
}

COUNTRY_MAPPING = {
    'Canada': 0, 'United Kingdom': 1, 'Italy': 2, 'United States': 3, 'Bulgaria': 4,
    'Netherlands': 5, 'Denmark': 6, 'Slovakia': 7, 'Ireland': 8, 'Norway': 9, 'Germany': 10,
    'Turkey': 11, 'Romania': 12, 'Azerbaijan': 13, 'France': 14, 'Belgium': 15,
    'Switzerland': 16, 'Greece': 17, 'Austria': 18, 'Hungary': 19, 'Portugal': 20,
    'Spain': 21, 'Serbia': 22, 'Czech Republic': 23, 'Sweden': 24, 'United Arab Emirates': 25,
    'Finland': 26, 'Estonia': 27, 'Poland': 28, 'Malta': 29, 'Belarus': 30, 'Liechtenstein': 31,
    'Bosnia and Herzegovina': 32, 'Cyprus': 33, 'Albania': 34, 'Croatia': 35,
    'Moldova, Republic of': 36, 'Slovenia': 37, 'Macedonia, The former Yugoslav Rep. of': 38,
    'Lithuania': 39, 'Luxembourg': 40, 'Georgia': 41, 'Latvia': 42, 'Montenegro': 43,
    'Iceland': 44, 'Monaco': 45, 'San Marino': 46, 'Andorra': 47
}

# New: continent mapping (maps a continent to a list of country names)
CONTINENT_MAPPING = {
    "North America": ['United States', 'Canada'],
    "Europe": [
        "Albania", "Andorra", "Austria", "Belgium", "Croatia", "Cyprus", "Czech Republic",
        "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
        "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
        "Portugal", "Romania", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland",
        "United Kingdom", "Turkey"
    ],
    "Other": ["United Arab Emirates"]
}