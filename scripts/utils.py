def rename_traffic_columns(df):
    """Rename Italian traffic dataset columns to English."""

    """
    YEAR_ID = survey year in format “aaaa”
    MONTH_ID = survey month in format “aaaamm”
    GIO_ID = survey day in format “aaaammgg”
    HHMI_ID = aggregation time per quarter hour in the format “hh: mm”
    APP_ID = MTS station number
    DIRMAR_COD = direction of travel, can take on the value 0 or 1
    VEI_ID = type of vehicle, can take the value from 0 to 10, the description of the type is present in the Vehicle Classes file
    NUM_TRANSITS = number of aggregate transits per quarter of an hour of the registered vehicle type

    """
    df = df.rename(columns={
        'ANNO_ID': 'Year',
        'MESE_ID': 'Month',
        'GIO_ID': 'Day',
        'HHMI_ID': 'HourMinute',
        'APP_ID': 'MTSStationID',
        'DIRMAR_COD': 'DirectionCode',
        'VEI_ID': 'VehicleType',
        'NUM_TRANSITI': 'TransitCount'
    })
    return df