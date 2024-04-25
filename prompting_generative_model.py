'''
DIFFERENT PROMPTS, PROMPT API CALLS, AND TESTS ARE COMMENTED THROUGHOUT THIS FILE.
IF YOU WISH TO RUN DIFFERENT EXPERIMENTS PLEASE COMMENT AND UNCOMMENT CODING BLOCKS ACCORDINGLY

ADDITIONALLY, YOU NEED TO PROVIDE YOUR OWN OPENAI API KEY
'''

import numpy as np
import math
from collections import OrderedDict
import openai

# jinja - for string formatting in prompts.

# Replace with your actual OpenAI API key
openai.api_key = 'INSERT YOUR API KEY HERE'

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


# VARIABLE ADJUSTABLE INPUT
# investor_preferences = {
#     "risk_tolerance": "moderate",
#     "time_horizon": "long-term",
#     "sector_preference": "Medicine"
# }

investor_preferences = {
    "risk_tolerance": "high",
    "time_horizon": "short-term",
    "sector_preference": "technology"
}


example_scenarios = [
    {
        "preferences": {
            "risk_tolerance": "high",
            "time_horizon": "short-term",
            "sector_preference": "technology"
        },
        "recommendations": ["AMD", "ANSS", "AMAT", "CDNS"]
    },
    {
        "preferences": {
            "risk_tolerance": "moderate",
            "time_horizon": "medium-term",
            "sector_preference": "financial"
        },
        "recommendations": ["MSCI"]
    },
    {
        "preferences": {
            "risk_tolerance": "low",
            "time_horizon": "long-term",
            "sector_preference": "consumer goods"
        },
        "recommendations": ["PG", "KDP"]
    },
    {
        "preferences": {
            "risk_tolerance": "very low",
            "time_horizon": "very long-term",
            "sector_preference": "utilities"
        },
        "recommendations": ["ATO", "ED", "WEC"]
    },
    {
        "preferences": {
            "risk_tolerance": "medium",
            "time_horizon": "long-term",
            "sector_preference": "healthcare"
        },
        "recommendations": ["ABT", "JNJ", "ISRG"]
    },
    {
        "preferences": {
            "risk_tolerance": "high",
            "time_horizon": "short-term",
            "sector_preference": "industrial"
        },
        "recommendations": ["EMR", "RTX"]
    },
    {
        "preferences": {
            "risk_tolerance": "moderate to high",
            "time_horizon": "medium to long-term",
            "sector_preference": "energy"
        },
        "recommendations": ["XOM", "CVX", "COP"]
    },
    {
        "preferences": {
            "risk_tolerance": "low to moderate",
            "time_horizon": "long-term",
            "sector_preference": "information technology"
        },
        "recommendations": ["ADI", "CSCO", "CTSH"]
    },
    {
        "preferences": {
            "risk_tolerance": "variable",
            "time_horizon": "short to medium-term",
            "sector_preference": "consumer services"
        },
        "recommendations": ["DIS"]
    },
    {
        "preferences": {
            "risk_tolerance": "moderate",
            "time_horizon": "medium-term",
            "sector_preference": "industrial"
        },
        "recommendations": ["EMR", "DOV", "ETN"]
    },
    {
        "preferences": {
            "risk_tolerance": "high",
            "time_horizon": "short-term",
            "sector_preference": "technology"
        },
        "recommendations": ["ADI", "MCHP", "SNPS"]
    },
    {
        "preferences": {
            "risk_tolerance": "very low",
            "time_horizon": "very long-term",
            "sector_preference": "utilities"
        },
        "recommendations": ["ATO", "ED", "WEC"]
    },
    {
        "preferences": {
            "risk_tolerance": "low",
            "time_horizon": "long-term",
            "sector_preference": "consumer goods"
        },
        "recommendations": ["PG"]
    },
    {
        "preferences": {
            "risk_tolerance": "moderate to high",
            "time_horizon": "medium-term",
            "sector_preference": "materials"
        },
        "recommendations": ["VMC", "MLM", "DD"]
    },
    {
        "preferences": {
            "risk_tolerance": "medium",
            "time_horizon": "medium-term",
            "sector_preference": "healthcare"
        },
        "recommendations": ["ABT", "BDX", "JNJ"]
    },
    {
        "preferences": {
            "risk_tolerance": "low to moderate",
            "time_horizon": "long-term",
            "sector_preference": "energy"
        },
        "recommendations": ["XOM", "CVX", "COP"]
    },
    {
        "preferences": {
            "risk_tolerance": "variable",
            "time_horizon": "short to medium-term",
            "sector_preference": "consumer services"
        },
        "recommendations": ["DIS"]
    },
    {
        "preferences": {
            "risk_tolerance": "high",
            "time_horizon": "short-term",
            "sector_preference": "information technology"
        },
        "recommendations": ["CRM", "CTSH", "TYL"]
    },
    {
        "preferences": {
            "risk_tolerance": "moderate",
            "time_horizon": "medium-term",
            "sector_preference": "financial"
        },
        "recommendations": ["MSCI"]
    }
]

# example_scenarios_rating = [
#     {
#         "preferences": {
#             "risk_tolerance": "high",
#             "time_horizon": "short-term",
#             "sector_preference": "technology"
#         },
#         "recommendations": {
#             "AMD": 8.2,
#             "ANSS": 6.0,
#             "AMAT": 5.6,
#             "CDNS": 5.0
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "moderate",
#             "time_horizon": "medium-term",
#             "sector_preference": "financial"
#         },
#         "recommendations": {
#             "MSCI": 7.5
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "low",
#             "time_horizon": "long-term",
#             "sector_preference": "consumer goods"
#         },
#         "recommendations": {
#             "PG": 7.8,
#             "KDP": 6.5
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "very low",
#             "time_horizon": "very long-term",
#             "sector_preference": "utilities"
#         },
#         "recommendations": {
#             "ATO": 8.0,
#             "ED": 8.5,
#             "WEC": 8.3
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "medium",
#             "time_horizon": "long-term",
#             "sector_preference": "healthcare"
#         },
#         "recommendations": {
#             "ABT": 7.2,
#             "JNJ": 7.9,
#             "ISRG": 6.8
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "high",
#             "time_horizon": "short-term",
#             "sector_preference": "industrial"
#         },
#         "recommendations": {
#             "EMR": 7.0,
#             "RTX": 6.5
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "moderate to high",
#             "time_horizon": "medium to long-term",
#             "sector_preference": "energy"
#         },
#         "recommendations": {
#             "XOM": 8.1,
#             "CVX": 7.7,
#             "COP": 7.3
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "low to moderate",
#             "time_horizon": "long-term",
#             "sector_preference": "information technology"
#         },
#         "recommendations": {
#             "ADI": 6.9,
#             "CSCO": 7.5,
#             "CTSH": 6.5
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "variable",
#             "time_horizon": "short to medium-term",
#             "sector_preference": "consumer services"
#         },
#         "recommendations": {
#             "DIS": 7.0
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "moderate",
#             "time_horizon": "medium-term",
#             "sector_preference": "industrial"
#         },
#         "recommendations": {
#             "EMR": 7.2,
#             "DOV": 6.8,
#             "ETN": 7.1
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "high",
#             "time_horizon": "short-term",
#             "sector_preference": "technology"
#         },
#         "recommendations": {
#             "ADI": 8.0,
#             "MCHP": 7.4,
#             "SNPS": 6.8
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "very low",
#             "time_horizon": "very long-term",
#             "sector_preference": "utilities"
#         },
#         "recommendations": {
#             "ATO": 8.6,
#             "ED": 8.8,
#             "WEC": 8.5
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "low",
#             "time_horizon": "long-term",
#             "sector_preference": "consumer goods"
#         },
#         "recommendations": {
#             "PG": 7.9
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "moderate to high",
#             "time_horizon": "medium-term",
#             "sector_preference": "materials"
#         },
#         "recommendations": {
#             "VMC": 7.0,
#             "MLM": 6.5,
#             "DD": 6.7
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "medium",
#             "time_horizon": "medium-term",
#             "sector_preference": "healthcare"
#         },
#         "recommendations": {
#             "ABT": 7.3,
#             "BDX": 7.6,
#             "JNJ": 7.9
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "low to moderate",
#             "time_horizon": "long-term",
#             "sector_preference": "energy"
#         },
#         "recommendations": {
#             "XOM": 7.5,
#             "CVX": 7.2,
#             "COP": 7.0
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "variable",
#             "time_horizon": "short to medium-term",
#             "sector_preference": "consumer services"
#         },
#         "recommendations": {
#             "DIS": 7.0
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "high",
#             "time_horizon": "short-term",
#             "sector_preference": "information technology"
#         },
#         "recommendations": {
#             "CRM": 6.5,
#             "CTSH": 6.8,
#             "TYL": 7.1
#         }
#     },
#     {
#         "preferences": {
#             "risk_tolerance": "moderate",
#             "time_horizon": "medium-term",
#             "sector_preference": "financial"
#         },
#         "recommendations": {
#             "MSCI": 7.5
#         }
#     }
# ]

number_of_recommendations = 4

prompt_text = f"""
You are an AI that provides investment recommendations based on an investor's preferences, including risk tolerance, time horizon, and sector preference. Here are some examples of how you have provided recommendations in the past:

{'. '.join([f'An investor with {scenario["preferences"]["risk_tolerance"]} risk tolerance, looking for {scenario["preferences"]["time_horizon"]} investments in the {scenario["preferences"]["sector_preference"]} sector was recommended: {", ".join(scenario["recommendations"])}' for scenario in example_scenarios])}.

Given the current economic conditions and considering the following companies list: {', '.join(sharia_companies)}.

Please analyse and provide a list of recommended companies for an investor with the following preferences:
Risk Tolerance: {investor_preferences["risk_tolerance"]}
Time Horizon: {investor_preferences["time_horizon"]}
Sector Preference: {investor_preferences["sector_preference"]}

You should recommend {number_of_recommendations} different stocks.

"""

"""THE FOLLOWING ARE EXAMPLE PROMPTS THAT EMULATE THE RANKING AND RATING TASKS RESPECTIVELY"""
# ranking_prompt_text = f"""
# You are an AI that provides investment recommendations based on an investor's preferences, including risk tolerance, time horizon, and sector preference. Here are some examples of how you have provided recommendations in the past:
#
# {'. '.join([f'An investor with {scenario["preferences"]["risk_tolerance"]} risk tolerance, looking for {scenario["preferences"]["time_horizon"]} investments in the {scenario["preferences"]["sector_preference"]} sector was recommended: {", ".join(scenario["recommendations"])}' for scenario in example_scenarios])}.
#
# For each example provided, the exemplar recommendations are listed from strongest to weakest. Meaning the first recommended stock is the most aligned for that corresponding investor. With the last recommended stock being the least aligned.
#
# Given the current economic conditions and considering the following companies list: {', '.join(sharia_companies)}.
#
# Please analyse and provide a list of recommended companies for an investor with the following preferences:
# Risk Tolerance: {investor_preferences["risk_tolerance"]}
# Time Horizon: {investor_preferences["time_horizon"]}
# Sector Preference: {investor_preferences["sector_preference"]}
#
# You should recommend {number_of_recommendations} different stocks.
#
# The recommendations should be ordered with the strongest recommendation first that seems most appropriate, all the way down to the last, the same way done in the examples.
# """
#
#
# rating_prompt_text = f"""
# You are an AI that provides investment recommendations based on an investor's preferences, including risk tolerance, time horizon, and sector preference. Here are some examples of how you have provided recommendations in the past:
#
# {'. '.join([f'An investor with {scenario["preferences"]["risk_tolerance"]} risk tolerance, looking for {scenario["preferences"]["time_horizon"]} investments in the {scenario["preferences"]["sector_preference"]} sector was recommended: {", ".join(scenario["recommendations"])}' for scenario in example_scenarios_rating])}.
#
# For each example provided, every exemplar recommendation has a score out of 10. This score represents the alignment of that stock with that current investor, representing its strength as an investment. With 10 being the highest score, and 1 being the lowest.
#
# Given the current economic conditions and considering the following companies list: {', '.join(sharia_companies)}.
#
# Please analyse and provide a list of recommended companies for an investor with the following preferences:
# Risk Tolerance: {investor_preferences["risk_tolerance"]}
# Time Horizon: {investor_preferences["time_horizon"]}
# Sector Preference: {investor_preferences["sector_preference"]}
#
# You should recommend {number_of_recommendations} different stocks.
#
# The recommendations should each have a score associated with them, representing the alignment and strength of the recommendation, the same way as done in the examples. Make your ratings as specific as possible, using specific decimals if necessary.
# """


# API Call
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant providing stock recommendations."},
        {"role": "user", "content": prompt_text}
    ],
    max_tokens=1000
)

"""RANKING AND RATING API CALLS"""
# response = openai.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant providing stock recommendations."},
#         {"role": "user", "content": ranking_prompt_text}
#     ],
#     max_tokens=1000
# )
#
# response = openai.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant providing stock recommendations."},
#         {"role": "user", "content": rating_prompt_text}
#     ],
#     max_tokens=1000
# )

# Accessing message content from the response
answer_content = response.choices[0].message.content

print("Generated Stock Recommendation and Explanation:")
print(answer_content)


# Then manually make a 'recommendations' array or dictionary by just reading the output from the model.
# Feed this back into the database - or in our case, we do this and use it for testing.


# PROMPT ORDERING TESTING
def calculate_global_entropy(recommendations):
    """
    recommendations: a list of recommended items for all users combined
    """
    item_freq = {}
    for item in recommendations:
        item_freq[item] = item_freq.get(item, 0) + 1

    total_items = len(recommendations)
    item_probs = {item: freq / total_items for item, freq in item_freq.items()}

    # Calculate entropy
    global_entropy = -sum(prob * math.log(prob, 2) for prob in item_probs.values())
    return global_entropy


def calculate_local_entropy(user_recommendations):
    """
    user_recommendations: a 2D Array i.e., list of lists with recommended items for each user
    """
    local_entropies = []

    for recommendations in user_recommendations:
        item_freq = {}
        for item in recommendations:
            item_freq[item] = item_freq.get(item, 0) + 1

        total_items = len(recommendations)
        item_probs = {item: freq / total_items for item, freq in item_freq.items()}

        # Calculate entropy
        local_entropy = -sum(prob * math.log(prob, 2) if prob > 0 else 0 for prob in item_probs.values())
        local_entropies.append(local_entropy)

    # In our case, average out the local entropies
    local_entropy = 0
    for result in local_entropies:
        local_entropy += result

    local_entropy = local_entropy / len(local_entropies)

    return local_entropy


# EXAMPLE USAGE (with random arrays for demonstration purposes)
# all_recommendations = ['Apple', 'Google', 'Amazon', 'Tesla', 'Microsoft', 'Apple']
#
# user1_recommendations = ['Apple', 'Google', 'Amazon']
# user2_recommendations = ['Apple', 'Microsoft', 'Amazon']
#
# global_entropy_score = calculate_global_entropy(all_recommendations)
# print(f"Global Entropy Score: {global_entropy_score}")
#
# local_entropies_scores = calculate_local_entropy([user1_recommendations, user2_recommendations])
# print(f"Local Entropies Scores: {local_entropies_scores}")




# RANKING VS RATING TESTING
def calculate_mrr_for_ranking(recommendations, relevant_stocks):
    """
    recommendations: a list of recommended items
    relevant_stocks: a list of desired items
    """
    for rank, stock in enumerate(recommendations, start=1):
        if stock == relevant_stocks[0]:
            return 1 / rank

    return 0


def calculate_mrr_for_rating(recommendations, relevant_stocks):
    """
    recommendations: a dictionary of recommended items with corresponding scores
    relevant_stocks: a dictionary of desired items with corresponding scores
    """
    max_score = ["", 0]
    for stock in relevant_stocks:
        score = relevant_stocks[stock]
        if score > max_score[1]:
            max_score = [stock, score]

    recommendations = {k: v for k, v in sorted(recommendations.items(), key=lambda item: item[1], reverse=True)}

    for rank, stock in enumerate(recommendations, start=1):
        if stock == max_score[0]:
            return 1 / rank

    return 0


def calculate_ndcg_for_ranking(recommendations, relevant_stocks):
    """
    recommendations: a list of recommended items
    relevant_stocks: a list of desired items
    """
    dcg = 0.0
    for i, stock in enumerate(recommendations, start=1):
        if stock in relevant_stocks:
            dcg += 1 / np.log2(i + 1)

    # Calculate ideal DCG, i.e., (DCG score of a perfectly ranked list)
    idcg = sum(1 / np.log2(i + 1) for i in range(1, len(relevant_stocks) + 1))

    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg


def calculate_ndcg_for_rating(recommended_items, relevant_stocks):
    """
    recommended_items: a dictionary with items as keys and their recommendation scores as values
    relevant_items_with_scores: a dictionary with relevant items as keys and their true scores as values
    """
    sorted_recommendations = sorted(recommended_items.items(), key=lambda item: item[1], reverse=True)

    # Calculate DCG
    dcg = 0.0
    for i, (item, score) in enumerate(sorted_recommendations, start=1):
        rel = relevant_stocks.get(item, 0)
        dcg += (2 ** rel - 1) / np.log2(i + 1)

    sorted_relevance = sorted(relevant_stocks.items(), key=lambda item: item[1], reverse=True)

    # Calculate ideal DCG, i.e., (DCG score of a perfectly ranked list)
    idcg = 0.0
    for i, (item, rel) in enumerate(sorted_relevance, start=1):
        idcg += (2 ** rel - 1) / np.log2(i + 1)

    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg


# EXAMPLE USAGE with random arrays for demonstration purposes:
# recommendations = ['Apple', 'Microsoft', 'Amazon']
# relevant_stocks = ['Microsoft', 'Tesla', 'Amazon']
# mrr_score = calculate_mrr_for_ranking(recommendations, relevant_stocks)
# print(f"Ranking MRR Score: {mrr_score}")
# ndcg_score = calculate_ndcg_for_ranking(recommendations, relevant_stocks)
# print(f"Ranking NDCG Score: {ndcg_score}")
#
# recommended_items = {"Apple": 0.49, "Microsoft": 0.6, "Amazon": 0.5}
# relevant_stocks = {"Tesla": 0.2, "Apple": 0.8}  # Assuming these are the true relevance scores
# mrr_score = calculate_mrr_for_rating(recommended_items, relevant_stocks)
# print(f"Rating MRR Score: {mrr_score}")
# ndcg_score = calculate_ndcg_for_rating(recommended_items, relevant_stocks)
# print(f"Rating NDCG Score: {ndcg_score}")
