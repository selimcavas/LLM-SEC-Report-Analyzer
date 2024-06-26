import re
from time import sleep
import requests
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sympy import li
from webdriver_manager.firefox import GeckoDriverManager
from bs4 import BeautifulSoup
import os


def extract_info_from_links():

    with open('earning_call_links.txt', 'r') as file:
        earning_call_links = file.read().splitlines()

    with open('earning_call_tickers.txt', 'w') as file:

        for link in earning_call_links:
            link = link.rstrip('/')

            # Split the URL into parts based on the '/' character
            parts = link.split('/')

            # The part we're interested in is the last part
            last_part = parts[-1]

            # Split the last part into subparts based on the '-' character
            subparts = last_part.split('-')

            # The ticker, quarter, and year are the first, second, and third subparts, respectively
            ticker = subparts[-6]
            quarter = subparts[-5]
            year = subparts[-4]

            print(f"Ticker: {ticker}, Quarter: {quarter}, Year: {year}")

            # Write the ticker, quarter, and year to the text file
            file.write(f"{ticker},{quarter},{year}\n")


def extract_earning_call_links():
    driver = webdriver.Firefox(
        service=FirefoxService(GeckoDriverManager().install()))

    # Open the text file containing the tickers
    with open('tickers.txt', 'r') as file:
        tickers = file.read().splitlines()

    # Base URL
    base_url = "https://www.fool.com/quote/nasdaq/"

    # Base XPath
    base_xpath_earning_call_a = "/html/body/div[9]/div[3]/div[1]/section[2]/div/div[1]/div[2]/div[8]/div/div/a["
    base_xpath_earning_call_b_plan = "/html/body/div[9]/div[3]/div[1]/section[2]/div/div[1]/div[2]/div[7]/div/div/a["

    # List to store the URLs
    urls = []

    # For each ticker, append it to the base URL and add the resulting URL to the list
    for ticker in tickers:
        url = base_url + ticker.lower() + "/#quote-earnings-transcripts"
        urls.append(url)

    with open('earning_call_links.txt', 'w') as file:
        # Print the URLs
        for index, url in enumerate(urls, start=1):
            print(f"Processing URL {index} of {len(urls)}")
            driver.get(url)
            sleep(3)
            # extracting earning call links
            if url == urls[0]:
                reject_cookies_xpath = """//*[@id="onetrust-reject-all-handler"]"""
                # click on the reject cookies button
                driver.find_element(By.XPATH, reject_cookies_xpath).click()

            for i in range(1, 5):
                # Construct the XPath string
                xpath_a = base_xpath_earning_call_a + str(i) + "]"
                xpath_b = base_xpath_earning_call_b_plan + str(i) + "]"

                # Wait for the element to be present and then select it
                try:
                    element = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, xpath_a)))
                except:
                    try:
                        element = WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located((By.XPATH, xpath_b)))
                    except:
                        print(
                            f"Element not found for index {i}. Skipping to next index.")
                        continue

                # Get the href attribute of the element
                link = element.get_attribute('href')

                # Write the link to the text file
                file.write(link + '\n')

    # Close the WebDriver
    driver.quit()


def scrape_and_save():
    with open('earning_call_tickers.txt', 'r') as tickers_file, open('earning_call_links.txt', 'r') as links_file:
        tickers = tickers_file.read().splitlines()
        links = links_file.read().splitlines()

    for ticker, link in zip(tickers, links):
        print(link)
        response = requests.get(link)
        soup = BeautifulSoup(response.content, 'html.parser')
        article_body = soup.find('div', class_='article-body')
        text = ' '.join(p.get_text() for p in article_body.find_all('p'))

        ticker, quarter, year = ticker.split(',')
        directory = f'data_collection/transcripts_sample/{ticker.upper()}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(f'{directory}/{ticker}_{quarter}_{year}.txt', 'w') as file:
            file.write(text)


def main():
    scrape_and_save()


if __name__ == "__main__":
    main()
