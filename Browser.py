from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import os
import time
from os.path import join
import platform


class Browser:
    def __init__(self):
        pathForDriver = self.findPathForDriver()
        self.__browser = webdriver.Chrome(executable_path=pathForDriver)

    def findPathForDriver(self):
        lookfor = "chromedriver.exe" if platform.system() == "Windows" else "chromedriver"
        placeWhereToStart = "C:\\" if platform.system() == "Windows" else "/"
        for root, dirs, files in os.walk(placeWhereToStart):
            if lookfor in files:
                return join(root, lookfor)

    def navigate_music(self, analyzedString):
        self.__browser.get("https://www.youtube.com/results?search_query="+analyzedString)
        time.sleep(1)