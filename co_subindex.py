
def get_CO_subindex(x):
    if pd.isna(x):  # Check if the value is NaN
        return 0
    if x <= 4.4:
        return x * 50 / 4.4
    elif x <= 9.4:
        return 50 + (x - 4.5) * (100 - 50) / (9.4 - 4.5)
    elif x <= 12.4:
        return 100 + (x - 9.5) * (150 - 100) / (12.4 - 9.5)
    elif x <= 15.4:
        return 150 + (x - 12.5) * (200 - 150) / (15.4 - 12.5)
    elif x <= 30.4:
        return 200 + (x - 15.5) * (300 - 200) / (30.4 - 15.5)
    elif x <= 50.4:
        return 300 + (x - 30.5) * (500 - 300) / (50.4 - 30.5)
    elif x > 50.4:
        return 500 + (x - 50.5) * (999 - 500) / (99999.9 - 50.4)
    else:
        return 0
