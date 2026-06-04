from jugaad_data.nse import derivative_history
from datetime import date
import pandas as pd

# 1. Define your parameters
symbol = "NIFTY"
strike_price = 25000
option_type = "CE"  # "CE" for Call, "PE" for Put
expiry_date = date(2026, 12, 31)  # The expiry date of the contract
from_date = date(2025, 12, 1)  # Start date for historical data
to_date = date(2025, 12, 31)  # End date for historical data

# 2. Fetch the data
# instrument_type is "OPTIDX" for Nifty/BankNifty options
try:
    df = derivative_history(
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
        expiry_date=expiry_date,
        strike_price=strike_price,
        option_type=option_type,
        instrument_type="OPTIDX"
    )

    # 3. Process and display
    if not df.empty:
        # Convert to Pandas DataFrame for easier analysis
        df = pd.DataFrame(df)

        # Keep relevant columns for Data Science analysis
        cols_to_keep = ['FH_TIMESTAMP', 'FH_OPEN_PRICE', 'FH_HIGH_PRICE',
                        'FH_LOW_PRICE', 'FH_CLOSING_PRICE', 'FH_TOT_TRD_QTY', 'FH_OPEN_INT']
        df = df[cols_to_keep]

        # Rename for clarity
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open_Interest']
        print(df.head())

        # Save to CSV for your project
        df.to_csv(f"nifty_{strike_price}_{option_type}_history.csv", index=False)
    else:
        print("No data found for the specified parameters.")

except Exception as e:
    print(f"An error occurred: {e}")