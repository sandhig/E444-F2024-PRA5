import requests
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt

def api_call(api_url, test_case):
    start_time = time.time()
    response = requests.post(api_url, json=test_case)
    end_time = time.time()
    latency = end_time - start_time
    return response.status_code, latency

def test():
    api_url = "http://pra5-take3.us-east-2.elasticbeanstalk.com/predict"
    
    test_articles = [
        "The sun sets in the east",  # Fake news
        "Scientists have found life on Mars",  # Fake news
        "There are 7 continents",  # Real news
        "Ontario is the capital of Canada"   # Real news
    ]
    
    results = []

    # 100 API calls for each test case
    for i in range(100): 
        for test_num, test_article in enumerate(test_articles):
            print(test_num,test_article)
            status_code, latency = api_call(api_url, test_article)
            results.append([test_num, test_article, latency])
            print(f"Test Case {test_num}, Article: {test_article}, Latency: {latency:.4f} seconds")
    
    # write results to CSV 
    with open("test_results.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Test Case", "Article", "Latency (seconds)"])  # Header
        writer.writerows(results)
    
    print("Latency results saved to 'test_results.csv'.")

    # generate boxplot
    generate_boxplot()

def generate_boxplot():
    df = pd.read_csv("test_results.csv")

    plt.figure(figsize=(10, 6))
    df.boxplot(column="Latency (seconds)", by="Test Case", grid=False)

    plt.title("Latency by Test Case")
    plt.suptitle("")
    plt.xlabel("Test Case")
    plt.ylabel("Latency (seconds)")

    plt.savefig("latency_boxplot.png")
    print("Boxplot saved to 'latency_boxplot.png'.")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    test()