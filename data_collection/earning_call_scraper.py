from time import sleep
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager

driver = webdriver.Firefox(
    service=FirefoxService(GeckoDriverManager().install()))

# Open the text file containing the tickers
with open('tickers.txt', 'r') as file:
    tickers = file.read().splitlines()

# Base URL
base_url = "https://www.fool.com/quote/nasdaq/"

# Base XPath
base_xpath_earning_call_a = "/html/body/div[8]/div[3]/div[1]/section[2]/div/div[1]/div[2]/div[8]/div/div/a["
base_xpath_earning_call_b_plan = "/html/body/div[8]/div[3]/div[1]/section[2]/div/div[1]/div[2]/div[7]/div/div/a["

# List to store the URLs
urls = []

# For each ticker, append it to the base URL and add the resulting URL to the list
for ticker in tickers:
    url = base_url + ticker.lower() + "/#quote-earnings-transcripts"
    urls.append(url)


with open('earning_call_links.txt', 'w') as file:
    # Print the URLs
    for url in urls:
        driver.get(url)

        sleep(3)

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