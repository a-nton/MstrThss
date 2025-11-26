#!/usr/bin/env python3
"""
Interactive Configuration Manager for GDELT Pipeline

Allows users to:
1. Select stocks from S&P 500 list
2. Configure date ranges
3. Modify config.py with selected settings
"""

import os
import re
from datetime import date, datetime

# S&P 500 stocks (Optimized for GDELT/News Search)
# Format: (Ticker, Search_Term)
SP500_STOCKS = [
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("GOOGL", "Google"),                # Changed from Alphabet (Google)
    ("AMZN", "Amazon"),
    ("NVDA", "NVIDIA"),
    ("META", "Meta"),         # "Meta" is too broad, keep full name?
    ("TSLA", "Tesla"),
    ("BRK.B", "Berkshire Hathaway"),
    ("UNH", "UnitedHealth"),            # Shortened for better hits
    ("JNJ", "Johnson & Johnson"),
    ("JPM", "JPMorgan"),                # Removed "Chase" for broader matching
    ("V", "Visa"),
    ("PG", "Procter & Gamble"),
    ("XOM", "Exxon Mobil"),
    ("MA", "Mastercard"),
    ("HD", "Home Depot"),
    ("CVX", "Chevron"),
    ("MRK", "Merck"),
    ("ABBV", "AbbVie"),
    ("PEP", "PepsiCo"),
    ("KO", "Coca-Cola"),
    ("COST", "Costco"),
    ("AVGO", "Broadcom"),
    ("LLY", "Eli Lilly"),
    ("WMT", "Walmart"),
    ("MCD", "McDonald's"),
    ("CSCO", "Cisco"),
    ("TMO", "Thermo Fisher"),           # Shortened
    ("ABT", "Abbott Laboratories"),
    ("ACN", "Accenture"),
    ("DIS", "Disney"),
    ("ADBE", "Adobe"),
    ("NFLX", "Netflix"),
    ("NKE", "Nike"),
    ("CRM", "Salesforce"),
    ("VZ", "Verizon"),
    ("CMCSA", "Comcast"),
    ("AMD", "AMD"),                     # "Advanced Micro Devices" is rare in headlines
    ("INTC", "Intel"),
    ("PFE", "Pfizer"),
    ("DHR", "Danaher"),
    ("TXN", "Texas Instruments"),
    ("UNP", "Union Pacific"),
    ("PM", "Philip Morris"),
    ("ORCL", "Oracle"),
    ("NEE", "NextEra Energy"),
    ("COP", "ConocoPhillips"),
    ("BMY", "Bristol Myers Squibb"),
    ("RTX", "Raytheon"),
    ("WFC", "Wells Fargo"),
]

def read_current_config():
    """Read current configuration from config.py"""
    config_path = "config.py"

    if not os.path.exists(config_path):
        return None, None, None, None

    with open(config_path, 'r') as f:
        content = f.read()

    # Extract ASSETS
    assets_match = re.search(r'ASSETS = \[(.*?)\]', content, re.DOTALL)
    current_assets = []
    if assets_match:
        assets_str = assets_match.group(1)
        for match in re.finditer(r'\{"symbol": "(\w+)", "name": "(.*?)"\}', assets_str):
            current_assets.append((match.group(1), match.group(2)))

    # Extract dates
    start_date = None
    end_date = None
    price_end_date = None

    start_match = re.search(r'DATA_START_DATE = date\((\d+), (\d+), (\d+)\)', content)
    if start_match:
        start_date = date(int(start_match.group(1)), int(start_match.group(2)), int(start_match.group(3)))

    end_match = re.search(r'NEWS_PREDICTION_END_DATE = date\((\d+), (\d+), (\d+)\)', content)
    if end_match:
        end_date = date(int(end_match.group(1)), int(end_match.group(2)), int(end_match.group(3)))

    price_match = re.search(r'PRICE_COLLECTION_END_DATE = date\((\d+), (\d+), (\d+)\)', content)
    if price_match:
        price_end_date = date(int(price_match.group(1)), int(price_match.group(2)), int(price_match.group(3)))

    return current_assets, start_date, end_date, price_end_date

def select_stocks(current_assets):
    """Interactive stock selection with search and multi-select"""
    print("\n" + "="*60)
    print("STOCK SELECTION (S&P 500)")
    print("="*60)

    # Show current selection
    if current_assets:
        print(f"\n📊 Currently selected: {', '.join([f'{t} ({n})' for t, n in current_assets])}")

    print("\nOptions:")
    print("1. Select stocks by number (browse all)")
    print("2. Search for stocks by name/ticker")
    print("3. Keep current selection")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "3":
        return current_assets

    elif choice == "2":
        # Search mode
        search_term = input("\nEnter search term (name or ticker): ").strip().upper()
        matches = [(t, n) for t, n in SP500_STOCKS if search_term in t.upper() or search_term in n.upper()]

        if not matches:
            print(f"❌ No stocks found matching '{search_term}'")
            return select_stocks(current_assets)

        print(f"\n✓ Found {len(matches)} matches:")
        for idx, (ticker, name) in enumerate(matches, 1):
            print(f"{idx:2d}. {ticker:6s} - {name}")

        print("\nEnter stock numbers to toggle (comma-separated, e.g., '1,3,5')")
        print("Or 'all' to select all, 'none' to clear, 'back' to go back")

        selection_input = input("Selection: ").strip()

        if selection_input.lower() == "back":
            return select_stocks(current_assets)
        elif selection_input.lower() == "all":
            return matches
        elif selection_input.lower() == "none":
            return []
        else:
            try:
                selected_indices = [int(x.strip()) for x in selection_input.split(",")]
                selected = [matches[i-1] for i in selected_indices if 0 < i <= len(matches)]
                return selected
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                return select_stocks(current_assets)

    else:
        # Browse mode - paginated
        page_size = 15
        selected_tickers = set([t for t, n in current_assets]) if current_assets else set()

        page = 0
        while True:
            start_idx = page * page_size
            end_idx = min(start_idx + page_size, len(SP500_STOCKS))

            print("\n" + "="*60)
            print(f"S&P 500 Stocks (Page {page+1}/{(len(SP500_STOCKS)-1)//page_size + 1})")
            print("="*60)

            for idx in range(start_idx, end_idx):
                ticker, name = SP500_STOCKS[idx]
                selected_mark = "✓" if ticker in selected_tickers else " "
                print(f"{idx+1:2d}. [{selected_mark}] {ticker:6s} - {name}")

            print("\n" + "-"*60)
            print("Commands:")
            print("  [number]       - Toggle stock selection")
            print("  [n]ext         - Next page")
            print("  [p]rev         - Previous page")
            print("  [d]one         - Finish selection")
            print("  [c]lear        - Clear all selections")
            print(f"\nCurrently selected: {len(selected_tickers)} stocks")

            cmd = input("\nCommand: ").strip().lower()

            if cmd == 'd' or cmd == 'done':
                # Convert selected tickers back to (ticker, name) tuples
                result = [(t, n) for t, n in SP500_STOCKS if t in selected_tickers]
                return result

            elif cmd == 'c' or cmd == 'clear':
                selected_tickers.clear()
                print("✓ Cleared all selections")

            elif cmd == 'n' or cmd == 'next':
                if end_idx < len(SP500_STOCKS):
                    page += 1

            elif cmd == 'p' or cmd == 'prev':
                if page > 0:
                    page -= 1

            else:
                # Try to parse as number
                try:
                    num = int(cmd)
                    if 1 <= num <= len(SP500_STOCKS):
                        ticker, name = SP500_STOCKS[num-1]
                        if ticker in selected_tickers:
                            selected_tickers.remove(ticker)
                            print(f"✓ Removed {ticker} - {name}")
                        else:
                            selected_tickers.add(ticker)
                            print(f"✓ Added {ticker} - {name}")
                    else:
                        print("❌ Invalid stock number")
                except ValueError:
                    print("❌ Invalid command")

def configure_dates(current_start, current_end, current_price_end):
    """Configure date ranges"""
    print("\n" + "="*60)
    print("DATE RANGE CONFIGURATION")
    print("="*60)

    if current_start and current_end:
        print(f"\n📅 Current settings:")
        print(f"  Analysis start:  {current_start}")
        print(f"  Analysis end:    {current_end}")
        print(f"  Price data end:  {current_price_end}")

    print("\nOptions:")
    print("1. Keep current dates")
    print("2. Configure new dates")

    choice = input("\nEnter choice (1-2): ").strip()

    if choice == "1":
        return current_start, current_end, current_price_end

    # Configure new dates
    print("\n" + "-"*60)
    print("Enter Analysis START date (when to begin analyzing news)")
    start_str = input("Format YYYY-MM-DD (e.g., 2025-09-01): ").strip()
    try:
        new_start = datetime.strptime(start_str, "%Y-%m-%d").date()
    except ValueError:
        print("❌ Invalid date format")
        return configure_dates(current_start, current_end, current_price_end)

    print("\n" + "-"*60)
    print("Enter Analysis END date (last day to analyze news)")
    end_str = input("Format YYYY-MM-DD (e.g., 2025-09-30): ").strip()
    try:
        new_end = datetime.strptime(end_str, "%Y-%m-%d").date()
    except ValueError:
        print("❌ Invalid date format")
        return configure_dates(current_start, current_end, current_price_end)

    if new_end <= new_start:
        print("❌ End date must be after start date")
        return configure_dates(current_start, current_end, current_price_end)

    # Calculate recommended price collection end date
    # Add ~30 days to allow for 1-month forward predictions
    from datetime import timedelta
    recommended_price_end = new_end + timedelta(days=30)

    print("\n" + "-"*60)
    print("Enter Price Collection END date (must be after analysis end)")
    print(f"Recommended: {recommended_price_end} (adds 30 days buffer for 1m predictions)")
    price_str = input(f"Format YYYY-MM-DD [press Enter for {recommended_price_end}]: ").strip()

    if not price_str:
        new_price_end = recommended_price_end
    else:
        try:
            new_price_end = datetime.strptime(price_str, "%Y-%m-%d").date()
        except ValueError:
            print("❌ Invalid date format")
            return configure_dates(current_start, current_end, current_price_end)

    if new_price_end <= new_end:
        print("❌ Price end date must be after analysis end date")
        return configure_dates(current_start, current_end, current_price_end)

    print("\n✓ Date configuration complete:")
    print(f"  Analysis: {new_start} to {new_end} ({(new_end - new_start).days + 1} days)")
    print(f"  Price data collected until: {new_price_end}")

    return new_start, new_end, new_price_end

def update_config_file(assets, start_date, end_date, price_end_date):
    """Update config.py with new settings"""
    config_path = "config.py"

    # Read current config
    with open(config_path, 'r') as f:
        content = f.read()

    # Update ASSETS
    assets_str = "ASSETS = [\n"
    for ticker, name in assets:
        assets_str += f'    {{"symbol": "{ticker}", "name": "{name}"}},\n'
    assets_str += "]"

    content = re.sub(
        r'ASSETS = \[.*?\]',
        assets_str,
        content,
        flags=re.DOTALL
    )

    # Update dates
    content = re.sub(
        r'DATA_START_DATE = date\(\d+, \d+, \d+\)',
        f'DATA_START_DATE = date({start_date.year}, {start_date.month}, {start_date.day})',
        content
    )

    content = re.sub(
        r'NEWS_PREDICTION_END_DATE = date\(\d+, \d+, \d+\)',
        f'NEWS_PREDICTION_END_DATE = date({end_date.year}, {end_date.month}, {end_date.day})',
        content
    )

    content = re.sub(
        r'PRICE_COLLECTION_END_DATE = date\(\d+, \d+, \d+\)',
        f'PRICE_COLLECTION_END_DATE = date({price_end_date.year}, {price_end_date.month}, {price_end_date.day})',
        content
    )

    # Write back
    with open(config_path, 'w') as f:
        f.write(content)

    print("\n✓ Configuration saved to config.py")

def main():
    print("="*60)
    print("       GDELT PIPELINE CONFIGURATION MANAGER")
    print("="*60)

    # Read current config
    current_assets, current_start, current_end, current_price_end = read_current_config()

    # Step 1: Select stocks
    selected_stocks = select_stocks(current_assets)

    if not selected_stocks:
        print("\n❌ No stocks selected. Exiting.")
        return

    # Step 2: Configure dates
    start_date, end_date, price_end_date = configure_dates(current_start, current_end, current_price_end)

    # Step 3: Confirm and save
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"\nStocks ({len(selected_stocks)}):")
    for ticker, name in selected_stocks:
        print(f"  • {ticker:6s} - {name}")

    print(f"\nDate Range:")
    print(f"  • Analysis Start:      {start_date}")
    print(f"  • Analysis End:        {end_date}")
    print(f"  • Price Data End:      {price_end_date}")
    print(f"  • Analysis Duration:   {(end_date - start_date).days + 1} days")

    confirm = input("\nSave this configuration? (yes/no): ").strip().lower()

    if confirm == "yes":
        update_config_file(selected_stocks, start_date, end_date, price_end_date)
        print("\n✅ Configuration updated successfully!")
    else:
        print("\n❌ Configuration not saved.")

if __name__ == "__main__":
    main()