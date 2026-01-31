import csv
import datetime
import os
import sys

def check_date_and_set_output(csv_path):
    """
    Checks if today's date exists in the provided CSV file (in 'fecha' column).
    If found, sets the 'should_run' output to 'true' for GitHub Actions.
    """
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    print(f"Checking for date: {today_str} in {csv_path}")

    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        sys.exit(1)

    found = False
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['fecha'] == today_str:
                    found = True
                    break
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if found:
        print(f"Match found for {today_str}. Setting should_run=true.")
        # Write to GITHUB_OUTPUT environment variable if it exists
        github_output = os.environ.get('GITHUB_OUTPUT')
        if github_output:
            with open(github_output, 'a') as f:
                f.write("should_run=true\n")
        else:
            print("GITHUB_OUTPUT not found, skipping output set.")
    else:
        print(f"No match found for {today_str}.")

if __name__ == "__main__":
    # Assuming the script is run from the root of the repo, or provide full path
    # Adapting to run from src/utils/ or root. 
    # If running from root, path is 'calendario.csv'
    # The user request implies checking 'calendario.csv' which is at root.
    file_path = 'calendario.csv'
    
    # Check if we need to adjust path (e.g. if script is run from src/utils/)
    # But usually in CI, we run from root.
    
    check_date_and_set_output(file_path)
