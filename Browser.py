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
        self.__pathForDriver = self.findPathForDriver()
        self.__active = False

    def __initBrowser(self):
        if not hasattr(self,"__browser") or self.__browser is None:
            self.__browser = webdriver.Chrome(executable_path=self.__pathForDriver)
            self.__active = True
        else:
            pass

    def is_active(self):
        return self.__active

    def findPathForDriver(self):
        lookfor = "chromedriver.exe" if platform.system() == "Windows" else "chromedriver"
        placeWhereToStart = "C:\\" if platform.system() == "Windows" else "/"
        for root, dirs, files in os.walk(placeWhereToStart):
            if lookfor in files:
                return join(root, lookfor)

    def navigate_music(self, analyzedString):
        if not hasattr(self,"__browser") or self.__browser is None:
            self.__initBrowser()
        self.__browser.get("https://www.youtube.com/results?search_query="+analyzedString)
        time.sleep(1)
        #TODO questo è l'Xpath "//ytd-video-renderer"[0]
        self.__browser.find_element_by_xpath("//ytd-video-renderer").click()


    def close(self):
        self.__browser.close()
        self.__browser = None
        self.__active = False

    def movement_detection(self,movement):
        if movement == "DX":
            self.__browser.find_element_by_xpath("//*[@class='ytp-next-button ytp-button']").click()
        else:
            self.__browser.execute_script("window.history.go(-1)")
        time.sleep(1)