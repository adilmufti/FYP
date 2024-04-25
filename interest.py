import requests
from bs4 import BeautifulSoup, NavigableString, Tag

# Scrape uses BeautifulSoup. Scrape could be used.
# Selenium automates the tasks that you use.

def find_interest_income(soup):
    rows = soup.find_all('div', {'data-test': 'fin-row'})
    for row in rows:
        title_div = row.find('div', title="Interest Income")
        if title_div:
            value_span = row.find('div', {'data-test': 'fin-col'}).find('span')
            if value_span:
                return value_span.text
    return None


def scrape_financial_data(ticker_symbol):
    interest_rev = None
    url = f"https://finance.yahoo.com/quote/{ticker_symbol}/financials"
    headers = {'User-Agent': 'Mozilla/5.0'}  # Including a user-agent

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "Error: Unable to fetch data"

    soup = BeautifulSoup(response.content, 'html.parser')

    # Replace these with the correct CSS selectors after inspecting the Yahoo Finance page
    total_revenue_selector = 'div[data-test="fin-col"] > span'
    interest_revenue_selector = 'div[data-test="fin-row"]:has(div[title="Interest Income after Provision for Loan Loss"]) div[data-test="fin-col"] > span'

    total_revenue = soup.select_one(total_revenue_selector).text

    if soup.select_one(interest_revenue_selector) == None:
        interest_revenue_selector = 'div[data-test="fin-row"]:has(div[title="Interest Income"]) div[data-test="fin-col"] > span'

    if soup.select_one(interest_revenue_selector) == None:
        interest_rev = find_interest_income(soup)

        if interest_rev == None or soup.select_one(interest_revenue_selector) == None:
            return {
                'total_revenue': total_revenue,
                'interest_revenue': '0',
            }

    interest_revenue = soup.select_one(interest_revenue_selector).text

    return {
        'total_revenue': total_revenue,
        'interest_revenue': interest_revenue
    }


def check_sharia_compliance(financial_data):
    # Remove commas and convert strings to integers
    total_revenue = int(financial_data['total_revenue'].replace(',', ''))
    interest_revenue = int(financial_data['interest_revenue'].replace(',', ''))

    # Calculate the percentage of interest revenue
    interest_percentage = round((interest_revenue / total_revenue) * 100, 1)

    # Check compliance (less than 5%)
    is_compliant = interest_percentage < 5
    return is_compliant, interest_percentage


# Example usage
"""
company = "BIO"
print("Company: ", company)

financial_data = scrape_financial_data(company)
print("Financial Data: ", financial_data)

compliance_status, interest_percentage = check_sharia_compliance(financial_data)
print(f"Are they Sharia Compliant: {compliance_status}")
print(f"Interest Percentage: {interest_percentage}% of total revenue comes from interest")
"""
