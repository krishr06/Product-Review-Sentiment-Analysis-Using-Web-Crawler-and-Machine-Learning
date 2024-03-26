import pandas as pd
import matplotlib.pyplot as plt
import joblib
from flask import Flask, render_template, request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

app = Flask(__name__)
colors = ['#00ff00', '#ff0000', '#ffff00']

def crawl_and_save(product_url, product_name):
    driver = webdriver.Chrome()
    driver.get(product_url)
    time.sleep(8)
    see_more_button = driver.find_element(By.XPATH, '//a[@data-hook="see-all-reviews-link-foot"]')
    see_more_button.click()
    try:
        sort_dropdown = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, 'sort-order-dropdown'))
        )
        select = Select(sort_dropdown)
        select.select_by_visible_text('Most recent')

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    reviews_data = []
    reviews_fetched = 0  

    while reviews_fetched < 100:  
        num_scrolls = 5
        for _ in range(num_scrolls):
            driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_DOWN)
            time.sleep(1)

        review_elements = driver.find_elements(By.CLASS_NAME, 'review-text')
        for review in review_elements:
            review_text = review.text.strip()
            
            reviews_data.append({'SrNo': len(reviews_data) + 1, 'Review': review_text})
            reviews_fetched += 1  
            if reviews_fetched == 100:
                break 

        print(f'Fetched {reviews_fetched} reviews.')

        if reviews_fetched == 100:
            break  

        next_page_button = driver.find_element(By.CLASS_NAME, 'a-last')
        if "a-disabled" in next_page_button.get_attribute("class") and "a-last" in next_page_button.get_attribute("class"):
            print('No more reviews available. Exiting.')
            break
        else:
            next_page_button.click()
            print(f'Navigating to the next page...')
            time.sleep(2)

    driver.quit()
    df = pd.DataFrame(reviews_data)
    #excel_file_name = f"reviews/{product_name}_reviews.xlsx"
    #df.to_excel(excel_file_name, index=False)
    #print("Crawling and saving data is complete.")

    perform_sentiment_analysis(df['Review'])


def perform_sentiment_analysis(reviews_data):
    loaded_model = joblib.load('random_forest_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    sentiments = []
    for review_text in reviews_data[:100]:  # Considering the first 100 reviews
        review_vectorized = vectorizer.transform([review_text])
        prediction = loaded_model.predict(review_vectorized)
        sentiment = 'Negative' if prediction == 1 else 'Positive'
        sentiments.append(sentiment)

    df_sentiments = pd.DataFrame({'Review': reviews_data[:100], 'Sentiment': sentiments})
    generate_pie_chart(df_sentiments)


def generate_pie_chart(df):
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['axes.facecolor'] = '#383D3B'

    plt.figure(figsize=(6, 6))
    plt.pie(df['Sentiment'].value_counts(), labels=df['Sentiment'].value_counts().index, autopct='%1.1f%%',colors=colors)
    plt.title("Sentiment Analysis")
    plt.savefig("static/pie_chart.png", facecolor='#383D3B')
    #plt.show()


@app.route('/')
def index():
    return render_template('index.html', img="")


@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    product_name = request.form['productName']
    product_url = request.form['productURL']    
    crawl_and_save(product_url, product_name)
    return render_template('index.html', img="static/pie_chart.png")


if __name__ == "__main__":
    app.run(debug=True)
