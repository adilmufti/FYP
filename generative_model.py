'''
TO RUN THIS CODE YOU NEED YOUR OWN OPENAI API KEY AND WILL NEED TO INSERT IT WHERE APPROPRIATE BELOW.

ALSO YOU WILL NEED YOUR OWN GPT-3.5 TURBO MODEL IDENTIFIER. DETAILS CAN BE FOUND BELOW:

ALSO YOU NEED YOUR OWN FINE-TUNED GPT-3.5 TURBO MODELS. THIS WILL REQUIRE YOU TO PUT SIGNIFICANT CREDIT INTO YOUR
OPENAI ACCOUNT.

IF YOU HAVE FINE-TUNED ON THE APPROPRIATE DATASETS FOUND IN THIS REPOSITORY, BY USING THE APPROPRIATE FILES
IN THIS REPOSITORY ('uploading_finet.py and fine-tune_job.py). THEN YOU WILL HAVE A SPECIAL MODEL CODE FOR YOUR
FINE-TUNED MODEL. YOU WILL FIND THIS UNIQUE CODE IDENTIFIER ON THE FINE-TUNING SECTION OF YOUR OPENAI DASHBOARD.

'''

from datasets import Dataset
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel, TrainingArguments, Trainer
import pandas as pd
import re
import openai
from openai import OpenAI
import numpy as np
from numpy.linalg import norm
import random


"""Company Dataset which was generated from running sharia_compliance.py"""
sharia_companies = ['AOS', 'ABT', 'AMD', 'ALB', 'AME', 'ADI', 'ANSS', 'AMAT', 'T', 'ATO', 'BKR', 'BDX', 'BIO', 'TECH',
                    'BWA', 'BSX', 'BLDR', 'CDNS', 'CVX', 'CTAS', 'CSCO', 'CTSH', 'COP', 'ED', 'CEG', 'COO', 'CPRT',
                    'GLW', 'CTVA', 'CTRA', 'CVS', 'DHR', 'DRI', 'DECK', 'XRAY', 'DVN', 'FANG', 'DG', 'DLTR', 'D', 'DOV',
                    'DOW', 'DHI', 'DD', 'EMN', 'ETN', 'EW', 'EMR', 'EOG', 'EQT', 'EQR', 'XOM', 'FAST', 'FDX', 'FTV',
                    'GRMN', 'GNRC', 'GD', 'GE', 'GIS', 'GPN', 'HAS', 'ILMN', 'INCY', 'IFF', 'ISRG', 'JBHT', 'J', 'JNJ',
                    'JCI', 'JNPR', 'KDP', 'KHC', 'LH', 'LEN', 'LIN', 'LKQ', 'LULU', 'MRO', 'MLM', 'MKC', 'META', 'MCHP',
                    'MU', 'MHK', 'MDLZ', 'MNST', 'MOS', 'MSCI', 'NEM', 'NWS', 'NDSN', 'NOC', 'PFE', 'PSX', 'PXD', 'PG',
                    'PLD', 'PHM', 'PWR', 'RTX', 'O', 'RMD', 'ROK', 'ROL', 'ROP', 'CRM', 'SLB', 'SRE', 'SWKS', 'SJM',
                    'SNA', 'SWK', 'STE', 'SNPS', 'TTWO', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TSCO', 'TT', 'TRMB',
                    'TYL', 'ULTA', 'VLO', 'VMC', 'WAB', 'DIS', 'WEC', 'WST', 'WRK', 'WY', 'ZBRA', 'ZBH']

similar_portfolios_database = {
    '4 2 Energy': ['CVX', 'COP', 'DVN'],
    '10 3 Finance': ['CSCO', 'J', 'JBHT'],
    '7 8 Finance': ['JBHT', 'J'],
    '2 9 Medicine': ['EW', 'ABT', 'MDLZ'],
    '9 2 Technology': ['TER', 'GOOGL', 'IBM'],
    '3 1 Medicine': ['EW', 'BDX', 'MDLZ'],
    '5 6 Energy': ['FANG', 'XOM'],
    '5 7 Consumer Goods': ['GIS', 'DIS', 'PG', 'CVS', 'UL'],
    '5 8 Medicine': ['JNJ', 'MDLZ', 'EW', 'ISRG'],
    '3 10 Finance': ['JBHT', 'VLO', 'J'],
    '3 3 Finance': ['VLO', 'CSCO', 'MSCI'],
    '2 4 Technology': ['IBM', 'SNPS', 'TEL', 'CRM'],
    '1 7 Energy': ['FANG', 'DVN', 'COP'],
    '3 4 Consumer Goods': ['PG', 'GIS', 'UL', 'DIS'],
    '1 4 Energy': ['FANG', 'CVX', 'EOG', 'DVN'],
    '2 3 Consumer Goods': ['PG', 'UL', 'KHC', 'CVS', 'GIS'],
    '4 5 Finance': ['TDY', 'CSCO', 'JBHT', 'VLO'],
    '10 7 Energy': ['CVX', 'EOG', 'XOM', 'FANG', 'DVN'],
    '1 1 Technology': ['META', 'AMD'],
    '8 10 Finance': ['CSCO', 'MSCI'],
    '3 7 Medicine': ['JNJ', 'BDX', 'ISRG', 'BIO', 'ABT'],
    '5 3 Consumer Goods': ['DIS', 'KHC', 'CVS', 'PG', 'UL'],
    '7 5 Technology': ['META', 'IBM', 'CTSH', 'CSCO', 'TER'],
    '10 6 Medicine': ['ABT', 'EW'],
    '6 3 Technology': ['SNPS', 'IBM', 'CSCO'],
    '3 9 Technology': ['AMD', 'TER', 'GOOGL', 'CRM', 'CTSH'],
    '7 4 Finance': ['MSCI', 'JBHT'],
    '10 9 Medicine': ['EW', 'JNJ', 'ISRG', 'BDX', 'ABT'],
    '5 9 Energy': ['DVN', 'CVX', 'COP', 'XOM', 'EOG'],
    '2 2 Finance': ['CSCO', 'JBHT', 'J', 'MSCI'],
    '5 4 Energy': ['DVN', 'FANG'],
    '2 3 Energy': ['CVX', 'XOM', 'FANG', 'COP', 'EOG'],
    '4 1 Technology': ['IBM', 'TEL', 'CRM', 'SNPS'],
    '5 1 Energy': ['DVN', 'FANG', 'COP', 'EOG', 'XOM'],
    '10 5 Technology': ['CTSH', 'CRM', 'META', 'IBM', 'GOOGL'],
    '5 1 Medicine': ['JNJ', 'ISRG'],
    '2 8 Consumer Goods': ['CVS', 'KHC', 'GIS'],
    '1 10 Energy': ['COP', 'XOM', 'DVN', 'FANG'],
    '7 3 Finance': ['CSCO', 'J'],
    '7 2 Energy': ['EOG', 'COP', 'XOM', 'CVX', 'DVN'],
    '3 4 Energy': ['CVX', 'DVN', 'COP'],
    '3 6 Consumer Goods': ['GIS', 'PG', 'KHC'],
    '10 2 Medicine': ['BDX', 'ILMN', 'EW'],
    '6 6 Finance': ['JBHT', 'VLO', 'CSCO', 'J', 'TDY'],
    '9 1 Medicine': ['ISRG', 'JNJ', 'BDX', 'BIO', 'EW'],
    '9 4 Medicine': ['ISRG', 'ILMN', 'MDLZ', 'JNJ', 'BIO'],
    '6 3 Finance': ['TDY', 'CSCO', 'J'],
    '4 6 Finance': ['CSCO', 'TDY', 'J', 'MSCI', 'VLO'],
    '1 1 Medicine': ['BDX', 'ISRG', 'MDLZ', 'EW', 'BIO'],
    '3 6 Finance': ['JBHT', 'TDY', 'CSCO', 'J', 'MSCI'],
    '8 8 Technology': ['CTSH', 'SNPS', 'GOOGL', 'IBM', 'CRM'],
    '1 8 Technology': ['CTSH', 'TER', 'CSCO', 'SNPS', 'IBM'],
    '8 10 Consumer Goods': ['UL', 'GIS', 'DIS', 'KHC'],
    '8 3 Consumer Goods': ['CVS', 'UL', 'KHC', 'PG'],
    '10 10 Consumer Goods': ['CVS', 'DIS', 'GIS'],
    '4 4 Energy': ['DVN', 'FANG'],
    '2 4 Consumer Goods': ['GIS', 'KHC', 'UL', 'PG'],
    '3 6 Energy': ['FANG', 'XOM', 'COP', 'EOG', 'DVN'],
    '5 3 Finance': ['J', 'MSCI', 'CSCO'],
    '7 10 Consumer Goods': ['CVS', 'GIS', 'DIS', 'KHC', 'UL'],
    '3 1 Finance': ['J', 'MSCI', 'CSCO'],
    '5 10 Consumer Goods': ['DIS', 'GIS', 'PG', 'KHC'],
    '5 5 Medicine': ['ISRG', 'MDLZ', 'ABT'],
    '6 3 Consumer Goods': ['GIS', 'PG', 'UL', 'KHC'],
    '2 1 Finance': ['JBHT', 'CSCO'],
    '7 1 Energy': ['EOG', 'FANG', 'DVN', 'COP'],
    '10 2 Finance': ['TDY', 'J', 'MSCI', 'VLO'],
    '10 8 Technology': ['IBM', 'AMD'],
    '5 2 Medicine': ['BDX', 'ISRG', 'JNJ'],
    '7 7 Finance': ['TDY', 'VLO', 'JBHT'],
    '10 3 Energy': ['FANG', 'COP'],
    '10 2 Energy': ['XOM', 'CVX', 'COP', 'EOG'],
    '7 1 Consumer Goods': ['UL', 'CVS', 'DIS'],
    '4 1 Energy': ['COP', 'DVN'],
    '7 10 Energy': ['XOM', 'DVN', 'CVX', 'COP']
}





"""
PART 1

- Filter out all non-Sharia compliant stocks from the snp500 data csv and then generate 
    a new, filtered, csv file with only Sharia compliant stocks
    
- Next we do dimensionality Reduction for LLM Input. So we extract financial figures from the stock data
    such as deviation in stock price. Then using these figures we generate a financial description of each
    company's stock. These short description is generated for all companies and passed into the LLM
"""
# Load the stock data
file_path = './snp500_data.csv'
stock_data = pd.read_csv(file_path)

stock_data.head()

stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# Filter out non-sharia compliant companies
filtered_data = stock_data[['Date'] + [col for col in stock_data.columns if col in sharia_companies]]
filtered_file_path = './filtered_snp500_data.csv'
filtered_data.to_csv(filtered_file_path, index=False)

file_path = './filtered_snp500_data.csv'
stock_data = pd.read_csv(file_path)

stock_data.head()

stock_data['Date'] = pd.to_datetime(stock_data['Date'])

stock_data.set_index('Date', inplace=True)

# Calculate descriptive statistics for each stock
descriptive_stats = stock_data.describe().transpose()

# Calculate price changes from the first to the last date for our financial descriptions
price_change = stock_data.iloc[-1] - stock_data.iloc[0]
price_change_percentage = (stock_data.iloc[-1] / stock_data.iloc[0] - 1) * 100


# Calculate monthly average returns for our financial descriptions
monthly_returns = stock_data.resample('M').mean().ffill().pct_change().mean() * 100

# Calculate volatility as the standard deviation of monthly returns for our financial descriptions
volatility = stock_data.resample('M').mean().ffill().pct_change().std() * 100


# Compile all these statistics into a single DataFrame
compiled_stats = pd.DataFrame({
    # 'MeanPrice': descriptive_stats['mean'],
    # 'MedianPrice': descriptive_stats['50%'],
    'StdDevPrice': descriptive_stats['std'],
    'PriceRange': descriptive_stats['max'] - descriptive_stats['min'],
    'PriceChange': price_change,
    'PriceChangePercentage': price_change_percentage,
    'MonthlyAvgReturn': monthly_returns,
    'Volatility': volatility
})

compiled_stats.head()

def create_comprehensive_stock_description(stats_df):
    descriptions = []
    for ticker, stats in stats_df.iterrows():
        description = (
            f"{ticker}: "
            # f"Mean price over period: ${stats['MeanPrice']:.2f}, "
            # f"Median price: ${stats['MedianPrice']:.2f}, "
            f"Price standard deviation: ${stats['StdDevPrice']:.2f}, "
            # f"Price range: ${stats['PriceRange']:.2f}, "
            # f"Absolute price change: ${stats['PriceChange']:.2f}, "
            f"Percentage price change: {stats['PriceChangePercentage']:.2f}%, "
            f"Average monthly return: {stats['MonthlyAvgReturn']:.2f}%, "
            f"Volatility: {stats['Volatility']:.2f}%."
        )
        descriptions.append(description)

    return " ".join(descriptions)


# Generate and print out the stock descriptions. we will manually copy and paste this into our model as input.
stock_descriptions = create_comprehensive_stock_description(compiled_stats)
print(stock_descriptions)





"""
PART 2

- Vector Similarity matching. So we take an investors preferences and use these to find the n closest
    pre-existing investment portfolios (where n is a adaptable variable). These portfolio's are then converted
    into a string format, and again are passed to the LLM as input

- Then we want to add the LLMs output back into the portfolio database. So we can grow and expand this 
    database to make the LLM iteratively learn from itself. So we have logic to convert the LLM output
    into suitable format for the database. 
    
"""
investor_preferences = ["3", "8", "Technology"]

def parse_vector(vector_str):
    """ Parse string into numerical vector and industry label. """
    parts = vector_str.split()
    return np.array([float(parts[0]), float(parts[1])]), parts[2]


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def find_similar_vectors(input_vector, store, k):
    """ Find the top k similar vectors in the dictionary matching the industry of the input vector. """
    input_vec, industry = parse_vector(" ".join(input_vector))
    similarities = []

    for key, value in store.items():
        vec, item_industry = parse_vector(key)
        if item_industry == industry:
            similarity = cosine_similarity(input_vec, vec)
            similarities.append((similarity, key, value))

    # Sort by similarity in descending order and return the key-value pairs of the top k items
    similarities.sort(reverse=True, key=lambda x: x[0])
    return [(key, value) for _, key, value in similarities[:k]]


k = 3
similar_indices = find_similar_vectors(investor_preferences, similar_portfolios_database, k)
print(similar_indices)



def extract_tickers(text):
    # Look for all capital words assuming they are stock tickers
    tickers = re.findall(r'\b[A-Z]{2,5}\b', text)
    return tickers

def create_portfolio(preferences, ticker):
    preferences_value = extract_tickers(ticker)
    preferences_key = " ".join(preferences)
    return preferences_key, preferences_value






"""
PART 3

- These functions are for testing our code from above, and are also for testing the LLM output, and extract results
    which were then manually scaled and standardised before being shown in the report.
"""

# TESTING
def get_stock_statistics_array(ticker, stats_df):
    if ticker in stats_df.index:
        stats = stats_df.loc[ticker]
        result = [
            float(f"{stats['StdDevPrice']:.2f}"),  # Price Standard Deviation
            float(f"{stats['PriceChangePercentage']:.2f}"),  # Percentage Price Change
            float(f"{stats['MonthlyAvgReturn']:.2f}"),  # Average Monthly Return
            float(f"{stats['Volatility']:.2f}")  # Volatility
        ]
        return result
    else:
        return f"No data found for ticker: {ticker}"


def calculate_difference_in_recommendations(recommended, target, stats_df):
    recommended_stats = []
    target_stats = []
    differences = []
    count = 0

    for stock in recommended:
        count += 1
        stock_stats = get_stock_statistics_array(stock, stats_df)

        for i in range(len(stock_stats)):
            recommended_stats.append(0)

        for i in range(len(stock_stats)):
            recommended_stats[i] += stock_stats[i]

    # Average Data Across All Stock Recommendations
    for entry in range(len(recommended_stats)):
        recommended_stats[entry] = recommended_stats[entry] / count

    count = 0
    for stock in target:
        count += 1
        stock_stats = get_stock_statistics_array(stock, stats_df)

        for i in range(len(stock_stats)):
            target_stats.append(0)

        for i in range(len(stock_stats)):
            target_stats[i] += stock_stats[i]

    # Average Data Across All Target Recommendations
    for entry in range(len(target_stats)):
        target_stats[entry] = target_stats[entry] / count

    # Calculate Difference
    for i in range(len(recommended_stats)):
        difference = abs(((recommended_stats[i] - target_stats[i]) / target_stats[i]) * 100)
        differences.append(round(difference, 2))

    # Take the average of distances. We can do this since we already evaluated consistency of the generation in
    # previous tests, so right now we can afford to average out the array and lose dimensionality and lose any
    # consistency depictions in the array. We just want an average now
    avg = 0
    for value in differences:
        avg += value

    avg = round((avg / len(differences)), 2)

    return avg


def create_stock_descriptions(tickers, stats_df):
    # Randomly select a subset of tickers for the description
    selected_tickers = random.sample(tickers, 5)  # Select 5 tickers for the description
    descriptions = []
    for ticker in selected_tickers:
        stats = stats_df.loc[ticker]
        description = (
            f"{ticker}: "
            f"Price standard deviation: ${stats['StdDevPrice']:.2f}, "
            f"Percentage price change: {stats['PriceChangePercentage']:.2f}%, "
            f"Average monthly return: {stats['MonthlyAvgReturn']:.2f}%, "
            f"Volatility: {stats['Volatility']:.2f}%."
        )
        descriptions.append(description)
    return " ".join(descriptions)


# def generate_synthetic_data(num_examples):
#     training_data = []
#     tickers = compiled_stats.index.tolist()  # List of all stock tickers
#     for _ in range(num_examples):
#         investor_prefs = get_investor_preferences()
#         stock_descriptions = create_stock_descriptions(tickers, compiled_stats)
#         recommended_stocks = random.sample(tickers, 3)  # Recommend 3 stocks
#
#         # Combine everything into a single text entry
#         text_entry = (
#             f"Investor preferences: {investor_prefs}. "
#             f"Stock data: {stock_descriptions} "
#             f"Recommended stocks: {', '.join(recommended_stocks)}."
#         )
#         training_data.append({"text": text_entry})
#     return training_data









api_key = 'INSERT YOUR OPEN AI API KEY HERE'

client = openai.OpenAI(api_key=api_key)


# completion = client.chat.completions.create(
#   model="INSERT YOUR UNIQUE FINE-TUNED MODEL IDENTIFIER HERE",
#   messages=[
#     {"role": "system", "content": "You are a stock recommender system, who will give stock recommendations to a user, given their specific investment preferences for time horizon, risk-tolerance and sector. To help evaluate these metrics, use the given stock information patterns as context."},
#     {"role": "user", "content": "Here are the list of all the stocks you can choose from these companies: 'AOS', 'ABT', 'AMD', 'ALB', 'AME', 'ADI', 'ANSS', 'AMAT', 'T', 'ATO', 'BKR', 'BDX', 'BIO', 'TECH', 'BWA', 'BSX', 'BLDR', 'CDNS', 'CVX', 'CTAS', 'CSCO', 'CTSH', 'COP', 'ED', 'CEG', 'COO', 'CPRT', 'GLW', 'CTVA', 'CTRA', 'CVS', 'DHR', 'DRI', 'DECK', 'XRAY', 'DVN', 'FANG', 'DG', 'DLTR', 'D', 'DOV', 'DOW', 'DHI', 'DD', 'EMN', 'ETN', 'EW', 'EMR', 'EOG', 'EQT', 'EQR', 'XOM', 'FAST', 'FDX', 'FTV', 'GRMN', 'GNRC', 'GD', 'GE', 'GIS', 'GPN', 'HAS', 'ILMN', 'INCY', 'IFF', 'ISRG', 'JBHT', 'J', 'JNJ', 'JCI', 'JNPR', 'KDP', 'KHC', 'LH', 'LEN', 'LIN', 'LKQ', 'LULU', 'MRO', 'MLM', 'MKC', 'META', 'MCHP', 'MU', 'MHK', 'MDLZ', 'MNST', 'MOS', 'MSCI', 'NEM', 'NWS', 'NDSN', 'NOC', 'PFE', 'PSX', 'PXD', 'PG', 'PLD', 'PHM', 'PWR', 'RTX', 'O', 'RMD', 'ROK', 'ROL', 'ROP', 'CRM', 'SLB', 'SRE', 'SWKS', 'SJM', 'SNA', 'SWK', 'STE', 'SNPS', 'TTWO', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TSCO', 'TT', 'TRMB', 'TYL', 'ULTA', 'VLO', 'VMC', 'WAB', 'DIS', 'WEC', 'WST', 'WRK', 'WY', 'ZBRA', 'ZBH']. And here is a breakdown of the financial patterns for each stock: ABT: Price standard deviation: $4.32, Percentage price change: 8.04%, Average monthly return: 1.39%, Volatility: 6.15%. ADI: Price standard deviation: $4.66, Percentage price change: 0.34%, Average monthly return: -0.47%, Volatility: 4.40%. ALB: Price standard deviation: $10.79, Percentage price change: 4.67%, Average monthly return: -1.95%, Volatility: 6.43%. AMAT: Price standard deviation: $13.55, Percentage price change: 40.49%, Average monthly return: 6.26%, Volatility: 9.75%. AMD: Price standard deviation: $6.12, Percentage price change: -19.35%, Average monthly return: -4.35%, Volatility: 5.79%. AME: Price standard deviation: $6.79, Percentage price change: 11.76%, Average monthly return: 3.22%, Volatility: 3.27%. ANSS: Price standard deviation: $25.11, Percentage price change: -10.20%, Average monthly return: -1.83%, Volatility: 12.48%. AOS: Price standard deviation: $5.16, Percentage price change: 28.86%, Average monthly return: 6.08%, Volatility: 2.37%. ATO: Price standard deviation: $5.42, Percentage price change: 6.14%, Average monthly return: 3.31%, Volatility: 3.93%. BDX: Price standard deviation: $6.95, Percentage price change: -3.07%, Average monthly return: -1.19%, Volatility: 3.73%. BIO: Price standard deviation: $27.96, Percentage price change: 2.24%, Average monthly return: 0.37%, Volatility: 7.57%. BKR: Price standard deviation: $1.59, Percentage price change: 23.80%, Average monthly return: 3.05%, Volatility: 12.73%. BSX: Price standard deviation: $2.28, Percentage price change: 18.43%, Average monthly return: 4.02%, Volatility: 2.46%. BWA: Price standard deviation: $3.24, Percentage price change: 34.92%, Average monthly return: 5.76%, Volatility: 3.18%. CDNS: Price standard deviation: $7.16, Percentage price change: -10.77%, Average monthly return: -1.56%, Volatility: 9.32%. CEG: Price standard deviation: $nan, Percentage price change: nan%, Average monthly return: nan%, Volatility: nan%. COO: Price standard deviation: $12.74, Percentage price change: 7.79%, Average monthly return: 1.72%, Volatility: 2.34%. COP: Price standard deviation: $4.84, Percentage price change: 47.25%, Average monthly return: 6.92%, Volatility: 9.95%. CPRT: Price standard deviation: $1.55, Percentage price change: 0.43%, Average monthly return: 1.48%, Volatility: 7.42%. CRM: Price standard deviation: $10.99, Percentage price change: -2.71%, Average monthly return: -0.20%, Volatility: 9.88%. CSCO: Price standard deviation: $2.93, Percentage price change: 21.27%, Average monthly return: 4.02%, Volatility: 2.44%. CTAS: Price standard deviation: $10.06, Percentage price change: 1.72%, Average monthly return: 1.47%, Volatility: 0.56%. CTRA: Price standard deviation: $0.72, Percentage price change: 8.74%, Average monthly return: -0.73%, Volatility: 5.17%. CTSH: Price standard deviation: $2.87, Percentage price change: -10.36%, Average monthly return: -1.48%, Volatility: 5.23%. CTVA: Price standard deviation: $2.64, Percentage price change: 20.06%, Average monthly return: 3.29%, Volatility: 2.99%. CVS: Price standard deviation: $4.01, Percentage price change: 28.29%, Average monthly return: 3.62%, Volatility: 6.01%. CVX: Price standard deviation: $7.23, Percentage price change: 28.67%, Average monthly return: 5.01%, Volatility: 7.04%. Given this information - here is a specific user's preferences ('risk_tolerance': 'low', 'time_horizon': 'long', 'sector_preference': 'technology')"}
#   ]
# )
completion = client.chat.completions.create(
  model="INSERT YOUR UNIQUE FINE-TUNED MODEL IDENTIFIER HERE",
  messages=[
    {"role": "system", "content": "You are a stock recommender system, who will give stock recommendations to a user, given their specific investment preferences for time horizon, risk-tolerance and sector. To help evaluate these metrics, use the given stock information patterns as context."},
    {"role": "user", "content": "Here are the list of all the stocks you can choose from these companies: 'AOS', 'ABT', 'AMD', 'ALB', 'AME', 'ADI', 'ANSS', 'AMAT', 'T', 'ATO', 'BKR', 'BDX', 'BIO', 'TECH', 'BWA', 'BSX', 'BLDR', 'CDNS', 'CVX', 'CTAS', 'CSCO', 'CTSH', 'COP', 'ED', 'CEG', 'COO', 'CPRT', 'GLW', 'CTVA', 'CTRA', 'CVS', 'DHR', 'DRI', 'DECK', 'XRAY', 'DVN', 'FANG', 'DG', 'DLTR', 'D', 'DOV', 'DOW', 'DHI', 'DD', 'EMN', 'ETN', 'EW', 'EMR', 'EOG', 'EQT', 'EQR', 'XOM', 'FAST', 'FDX', 'FTV', 'GRMN', 'GNRC', 'GD', 'GE', 'GIS', 'GPN', 'HAS', 'ILMN', 'INCY', 'IFF', 'ISRG', 'JBHT', 'J', 'JNJ', 'JCI', 'JNPR', 'KDP', 'KHC', 'LH', 'LEN', 'LIN', 'LKQ', 'LULU', 'MRO', 'MLM', 'MKC', 'META', 'MCHP', 'MU', 'MHK', 'MDLZ', 'MNST', 'MOS', 'MSCI', 'NEM', 'NWS', 'NDSN', 'NOC', 'PFE', 'PSX', 'PXD', 'PG', 'PLD', 'PHM', 'PWR', 'RTX', 'O', 'RMD', 'ROK', 'ROL', 'ROP', 'CRM', 'SLB', 'SRE', 'SWKS', 'SJM', 'SNA', 'SWK', 'STE', 'SNPS', 'TTWO', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TSCO', 'TT', 'TRMB', 'TYL', 'ULTA', 'VLO', 'VMC', 'WAB', 'DIS', 'WEC', 'WST', 'WRK', 'WY', 'ZBRA', 'ZBH']. And here is a breakdown of the financial patterns for each stock: ABT: Price standard deviation: $4.32, Percentage price change: 8.04%, Average monthly return: 1.39%, Volatility: 6.15%. ADI: Price standard deviation: $4.66, Percentage price change: 0.34%, Average monthly return: -0.47%, Volatility: 4.40%. ALB: Price standard deviation: $10.79, Percentage price change: 4.67%, Average monthly return: -1.95%, Volatility: 6.43%. AMAT: Price standard deviation: $13.55, Percentage price change: 40.49%, Average monthly return: 6.26%, Volatility: 9.75%. AMD: Price standard deviation: $6.12, Percentage price change: -19.35%, Average monthly return: -4.35%, Volatility: 5.79%. AME: Price standard deviation: $6.79, Percentage price change: 11.76%, Average monthly return: 3.22%, Volatility: 3.27%. ANSS: Price standard deviation: $25.11, Percentage price change: -10.20%, Average monthly return: -1.83%, Volatility: 12.48%. AOS: Price standard deviation: $5.16, Percentage price change: 28.86%, Average monthly return: 6.08%, Volatility: 2.37%. ATO: Price standard deviation: $5.42, Percentage price change: 6.14%, Average monthly return: 3.31%, Volatility: 3.93%. BDX: Price standard deviation: $6.95, Percentage price change: -3.07%, Average monthly return: -1.19%, Volatility: 3.73%. BIO: Price standard deviation: $27.96, Percentage price change: 2.24%, Average monthly return: 0.37%, Volatility: 7.57%. BKR: Price standard deviation: $1.59, Percentage price change: 23.80%, Average monthly return: 3.05%, Volatility: 12.73%. BSX: Price standard deviation: $2.28, Percentage price change: 18.43%, Average monthly return: 4.02%, Volatility: 2.46%. BWA: Price standard deviation: $3.24, Percentage price change: 34.92%, Average monthly return: 5.76%, Volatility: 3.18%. CDNS: Price standard deviation: $7.16, Percentage price change: -10.77%, Average monthly return: -1.56%, Volatility: 9.32%. CEG: Price standard deviation: $nan, Percentage price change: nan%, Average monthly return: nan%, Volatility: nan%. COO: Price standard deviation: $12.74, Percentage price change: 7.79%, Average monthly return: 1.72%, Volatility: 2.34%. COP: Price standard deviation: $4.84, Percentage price change: 47.25%, Average monthly return: 6.92%, Volatility: 9.95%. CPRT: Price standard deviation: $1.55, Percentage price change: 0.43%, Average monthly return: 1.48%, Volatility: 7.42%. CRM: Price standard deviation: $10.99, Percentage price change: -2.71%, Average monthly return: -0.20%, Volatility: 9.88%. CSCO: Price standard deviation: $2.93, Percentage price change: 21.27%, Average monthly return: 4.02%, Volatility: 2.44%. CTAS: Price standard deviation: $10.06, Percentage price change: 1.72%, Average monthly return: 1.47%, Volatility: 0.56%. CTRA: Price standard deviation: $0.72, Percentage price change: 8.74%, Average monthly return: -0.73%, Volatility: 5.17%. CTSH: Price standard deviation: $2.87, Percentage price change: -10.36%, Average monthly return: -1.48%, Volatility: 5.23%. CTVA: Price standard deviation: $2.64, Percentage price change: 20.06%, Average monthly return: 3.29%, Volatility: 2.99%. CVS: Price standard deviation: $4.01, Percentage price change: 28.29%, Average monthly return: 3.62%, Volatility: 6.01%. CVX: Price standard deviation: $7.23, Percentage price change: 28.67%, Average monthly return: 5.01%, Volatility: 7.04%. D: Price standard deviation: $2.98, Percentage price change: 5.38%, Average monthly return: 2.22%, Volatility: 3.25%. DD: Price standard deviation: $3.92, Percentage price change: 21.56%, Average monthly return: 0.73%, Volatility: 8.17%. DG: Price standard deviation: $10.55, Percentage price change: -2.07%, Average monthly return: 0.62%, Volatility: 7.37%. DHI: Price standard deviation: $10.08, Percentage price change: 37.58%, Average monthly return: 8.48%, Volatility: 5.05%. DHR: Price standard deviation: $11.84, Percentage price change: 12.08%, Average monthly return: 2.19%, Volatility: 7.32%. DIS: Price standard deviation: $7.95, Percentage price change: -4.50%, Average monthly return: 0.89%, Volatility: 5.07%. DLTR: Price standard deviation: $5.29, Percentage price change: 2.89%, Average monthly return: 1.13%, Volatility: 4.94%. DOV: Price standard deviation: $10.51, Percentage price change: 20.31%, Average monthly return: 4.97%, Volatility: 5.60%. DOW: Price standard deviation: $3.87, Percentage price change: 30.09%, Average monthly return: 4.91%, Volatility: 4.39%. DRI: Price standard deviation: $8.68, Percentage price change: 21.23%, Average monthly return: 3.84%, Volatility: 5.17%. DVN: Price standard deviation: $2.30, Percentage price change: 64.68%, Average monthly return: 9.32%, Volatility: 9.12%. ED: Price standard deviation: $3.66, Percentage price change: 12.34%, Average monthly return: 3.26%, Volatility: 3.00%. EMN: Price standard deviation: $6.55, Percentage price change: 29.63%, Average monthly return: 4.86%, Volatility: 4.14%. EMR: Price standard deviation: $4.34, Percentage price change: 22.16%, Average monthly return: 3.63%, Volatility: 1.69%. EOG: Price standard deviation: $7.55, Percentage price change: 68.38%, Average monthly return: 9.52%, Volatility: 7.25%. EQR: Price standard deviation: $5.03, Percentage price change: 30.64%, Average monthly return: 5.46%, Volatility: 4.31%. EQT: Price standard deviation: $1.60, Percentage price change: 69.43%, Average monthly return: 6.88%, Volatility: 7.52%. ETN: Price standard deviation: $8.95, Percentage price change: 22.85%, Average monthly return: 4.52%, Volatility: 3.83%. EW: Price standard deviation: $4.42, Percentage price change: 1.29%, Average monthly return: 1.10%, Volatility: 6.28%. FANG: Price standard deviation: $8.13, Percentage price change: 65.22%, Average monthly return: 8.74%, Volatility: 8.03%. FAST: Price standard deviation: $2.42, Percentage price change: 11.24%, Average monthly return: 2.48%, Volatility: 4.68%. FDX: Price standard deviation: $19.59, Percentage price change: 21.19%, Average monthly return: 5.39%, Volatility: 2.55%. FTV: Price standard deviation: $2.31, Percentage price change: 0.73%, Average monthly return: 0.56%, Volatility: 3.88%. GD: Price standard deviation: $14.32, Percentage price change: 31.53%, Average monthly return: 6.36%, Volatility: 1.98%. GE: Price standard deviation: $5.97, Percentage price change: 23.97%, Average monthly return: 4.17%, Volatility: 5.68%. GIS: Price standard deviation: $2.56, Percentage price change: 8.44%, Average monthly return: 2.70%, Volatility: 2.11%. GLW: Price standard deviation: $3.42, Percentage price change: 25.00%, Average monthly return: 5.02%, Volatility: 6.23%. GNRC: Price standard deviation: $31.75, Percentage price change: 31.75%, Average monthly return: 5.16%, Volatility: 11.18%. GPN: Price standard deviation: $9.63, Percentage price change: -5.54%, Average monthly return: 0.85%, Volatility: 5.39%. GRMN: Price standard deviation: $7.48, Percentage price change: 18.64%, Average monthly return: 3.89%, Volatility: 3.75%. HAS: Price standard deviation: $2.42, Percentage price change: 4.76%, Average monthly return: 0.83%, Volatility: 2.64%. IFF: Price standard deviation: $9.53, Percentage price change: 35.92%, Average monthly return: 5.45%, Volatility: 6.45%. ILMN: Price standard deviation: $31.65, Percentage price change: 5.06%, Average monthly return: -0.33%, Volatility: 11.98%. INCY: Price standard deviation: $5.57, Percentage price change: -5.40%, Average monthly return: -2.79%, Volatility: 5.79%. ISRG: Price standard deviation: $16.16, Percentage price change: 1.83%, Average monthly return: 1.79%, Volatility: 8.30%. J: Price standard deviation: $11.28, Percentage price change: 27.99%, Average monthly return: 5.98%, Volatility: 4.12%. JBHT: Price standard deviation: $12.40, Percentage price change: 29.10%, Average monthly return: 5.06%, Volatility: 4.21%. JCI: Price standard deviation: $5.01, Percentage price change: 38.83%, Average monthly return: 6.42%, Volatility: 3.71%. JNJ: Price standard deviation: $3.50, Percentage price change: 9.59%, Average monthly return: 1.19%, Volatility: 2.10%. JNPR: Price standard deviation: $0.98, Percentage price change: 17.31%, Average monthly return: 2.16%, Volatility: 1.78%. KDP: Price standard deviation: $1.86, Percentage price change: 14.99%, Average monthly return: 3.27%, Volatility: 3.40%. KHC: Price standard deviation: $3.31, Percentage price change: 28.93%, Average monthly return: 7.06%, Volatility: 2.25%. LEN: Price standard deviation: $9.87, Percentage price change: 29.34%, Average monthly return: 6.96%, Volatility: 6.03%. LH: Price standard deviation: $15.25, Percentage price change: 30.33%, Average monthly return: 5.57%, Volatility: 1.97%. LIN: Price standard deviation: $16.97, Percentage price change: 15.64%, Average monthly return: 3.55%, Volatility: 4.68%. LKQ: Price standard deviation: $4.02, Percentage price change: 43.63%, Average monthly return: 7.32%, Volatility: 4.59%. LULU: Price standard deviation: $18.31, Percentage price change: -12.45%, Average monthly return: -2.07%, Volatility: 5.63%. MCHP: Price standard deviation: $3.35, Percentage price change: 4.09%, Average monthly return: -0.02%, Volatility: 6.40%. MDLZ: Price standard deviation: $2.34, Percentage price change: 7.92%, Average monthly return: 2.17%, Volatility: 3.85%. META: Price standard deviation: $21.85, Percentage price change: 15.25%, Average monthly return: 4.38%, Volatility: 5.02%. MHK: Price standard deviation: $25.69, Percentage price change: 57.73%, Average monthly return: 10.73%, Volatility: 2.42%. MKC: Price standard deviation: $2.84, Percentage price change: -4.75%, Average monthly return: -0.73%, Volatility: 4.03%. MLM: Price standard deviation: $24.87, Percentage price change: 34.33%, Average monthly return: 5.57%, Volatility: 1.36%. MNST: Price standard deviation: $1.74, Percentage price change: 0.23%, Average monthly return: 0.69%, Volatility: 5.43%. MOS: Price standard deviation: $3.00, Percentage price change: 54.32%, Average monthly return: 6.94%, Volatility: 2.80%. MRO: Price standard deviation: $1.58, Percentage price change: 75.83%, Average monthly return: 10.90%, Volatility: 12.56%. MSCI: Price standard deviation: $26.36, Percentage price change: 4.12%, Average monthly return: 2.97%, Volatility: 6.09%. MU: Price standard deviation: $5.13, Percentage price change: 6.52%, Average monthly return: 0.82%, Volatility: 7.33%. NDSN: Price standard deviation: $8.76, Percentage price change: 2.31%, Average monthly return: 1.54%, Volatility: 4.55%. NEM: Price standard deviation: $3.65, Percentage price change: 18.93%, Average monthly return: 2.80%, Volatility: 6.64%. NOC: Price standard deviation: $25.73, Percentage price change: 25.66%, Average monthly return: 5.88%, Volatility: 4.32%. NWS: Price standard deviation: $2.49, Percentage price change: 39.26%, Average monthly return: 7.90%, Volatility: 8.52%. O: Price standard deviation: $3.08, Percentage price change: 11.01%, Average monthly return: 3.38%, Volatility: 3.72%. PFE: Price standard deviation: $1.78, Percentage price change: 11.07%, Average monthly return: 2.54%, Volatility: 5.62%. PG: Price standard deviation: $4.18, Percentage price change: 0.14%, Average monthly return: 0.61%, Volatility: 3.85%. PHM: Price standard deviation: $5.43, Percentage price change: 33.36%, Average monthly return: 7.79%, Volatility: 3.76%. PLD: Price standard deviation: $6.42, Percentage price change: 20.46%, Average monthly return: 4.13%, Volatility: 6.27%. PSX: Price standard deviation: $5.41, Percentage price change: 25.57%, Average monthly return: 4.88%, Volatility: 7.59%. PWR: Price standard deviation: $8.92, Percentage price change: 36.63%, Average monthly return: 6.86%, Volatility: 4.38%. PXD: Price standard deviation: $12.28, Percentage price change: 40.04%, Average monthly return: 6.38%, Volatility: 10.37%. RMD: Price standard deviation: $10.09, Percentage price change: -8.42%, Average monthly return: -2.36%, Volatility: 6.94%. ROK: Price standard deviation: $8.60, Percentage price change: 6.95%, Average monthly return: 1.02%, Volatility: 4.13%. ROL: Price standard deviation: $1.93, Percentage price change: -7.03%, Average monthly return: -0.65%, Volatility: 6.74%. ROP: Price standard deviation: $20.55, Percentage price change: 3.44%, Average monthly return: 1.43%, Volatility: 6.31%. RTX: Price standard deviation: $5.16, Percentage price change: 25.54%, Average monthly return: 5.49%, Volatility: 1.65%. SJM: Price standard deviation: $7.79, Percentage price change: 17.33%, Average monthly return: 4.13%, Volatility: 3.71%. SLB: Price standard deviation: $2.46, Percentage price change: 51.37%, Average monthly return: 7.17%, Volatility: 9.50%. SNA: Price standard deviation: $25.79, Percentage price change: 50.87%, Average monthly return: 9.51%, Volatility: 4.43%. SNPS: Price standard deviation: $16.07, Percentage price change: -7.72%, Average monthly return: -2.12%, Volatility: 9.88%. SRE: Price standard deviation: $3.32, Percentage price change: 12.86%, Average monthly return: 3.49%, Volatility: 3.43%. STE: Price standard deviation: $11.68, Percentage price change: 4.20%, Average monthly return: 1.86%, Volatility: 7.38%. SWK: Price standard deviation: $14.59, Percentage price change: 25.58%, Average monthly return: 5.43%, Volatility: 4.39%. SWKS: Price standard deviation: $11.46, Percentage price change: 8.74%, Average monthly return: 1.92%, Volatility: 12.07%. T: Price standard deviation: $0.85, Percentage price change: 5.08%, Average monthly return: 3.00%, Volatility: 2.41%. TDY: Price standard deviation: $23.70, Percentage price change: 12.98%, Average monthly return: 3.09%, Volatility: 5.05%. TECH: Price standard deviation: $7.91, Percentage price change: 27.34%, Average monthly return: 5.32%, Volatility: 8.56%. TEL: Price standard deviation: $3.76, Percentage price change: 9.51%, Average monthly return: 1.48%, Volatility: 0.57%. TER: Price standard deviation: $8.86, Percentage price change: -0.28%, Average monthly return: -1.58%, Volatility: 10.13%. TFX: Price standard deviation: $16.15, Percentage price change: -2.22%, Average monthly return: 0.76%, Volatility: 4.94%. TRMB: Price standard deviation: $4.76, Percentage price change: 13.83%, Average monthly return: 2.84%, Volatility: 6.85%. TSCO: Price standard deviation: $14.98, Percentage price change: 32.79%, Average monthly return: 6.10%, Volatility: 1.92%. TSLA: Price standard deviation: $28.98, Percentage price change: -20.81%, Average monthly return: -6.29%, Volatility: 10.97%. TT: Price standard deviation: $11.52, Percentage price change: 24.96%, Average monthly return: 5.02%, Volatility: 3.63%. TTWO: Price standard deviation: $14.15, Percentage price change: -16.92%, Average monthly return: -4.39%, Volatility: 6.84%. TYL: Price standard deviation: $22.23, Percentage price change: -8.11%, Average monthly return: -1.46%, Volatility: 8.03%. ULTA: Price standard deviation: $16.05, Percentage price change: 15.72%, Average monthly return: 2.27%, Volatility: 3.88%. VLO: Price standard deviation: $7.63, Percentage price change: 46.83%, Average monthly return: 8.80%, Volatility: 9.75%. VMC: Price standard deviation: $11.42, Percentage price change: 29.99%, Average monthly return: 4.95%, Volatility: 2.69%. WAB: Price standard deviation: $3.35, Percentage price change: 9.21%, Average monthly return: 0.47%, Volatility: 4.41%. WEC: Price standard deviation: $4.82, Percentage price change: 7.16%, Average monthly return: 2.62%, Volatility: 5.26%. WRK: Price standard deviation: $5.36, Percentage price change: 41.88%, Average monthly return: 7.76%, Volatility: 8.72%. WST: Price standard deviation: $19.87, Percentage price change: 14.59%, Average monthly return: 2.80%, Volatility: 8.49%. WY: Price standard deviation: $2.13, Percentage price change: 14.33%, Average monthly return: 4.42%, Volatility: 3.82%. XOM: Price standard deviation: $5.13, Percentage price change: 50.24%, Average monthly return: 7.70%, Volatility: 7.45%. XRAY: Price standard deviation: $4.77, Percentage price change: 25.64%, Average monthly return: 4.53%, Volatility: 7.69%. ZBH: Price standard deviation: $6.36, Percentage price change: 8.90%, Average monthly return: 1.77%, Volatility: 3.80%. ZBRA: Price standard deviation: $39.34, Percentage price change: 26.29%, Average monthly return: 4.71%, Volatility: 7.27%. Given this information - here is a specific user's preferences ('risk_tolerance': '8', 'time_horizon': '2', 'sector_preference': 'technology'). If you require further guidance - For additional context, other previous investors with similar preferences made the following investments: ('risk: 3, time-horizion: 9, sector preference: Technology, invested in: 'AMD', 'TER', 'GOOGL', 'CRM', 'CTSH'. One with risk: 2, time-horizion: 4, sector preference: Technology, invested in: 'IBM', 'SNPS', 'TEL', 'CRM'.  One with risk: 1, time-horizion: 8, sector preference: Technology, invested in: 'CTSH', 'TER', 'CSCO', 'SNPS', 'IBM'"}
  ]
)
model_output = completion.choices[0].message.content
print(completion.choices[0].message.content)


# Step 2 - Adding generated portfolio to database
portfolio_key, portfolio_value = create_portfolio(investor_preferences, model_output)
similar_portfolios_database[portfolio_key] = portfolio_value













