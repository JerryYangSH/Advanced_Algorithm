#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

problemCount = 0
OPTION = False

def getLineRange(driver):
    lineNum = driver.find_element_by_css_selector("div.ace_gutter")
    lineNumStrList = lineNum.text.split('\n')
    startLine = int(lineNumStrList[0])
    endLine = int(lineNumStrList[-1])
    pageLines = endLine - startLine + 1
    return startLine, endLine, pageLines

def merge(sourcecodeList, preStart, preEnd,
          nowcodeList,    start,    end):
    print "Merging len {} and {}".format(len(sourcecodeList), len(nowcodeList))
    resultList = sourcecodeList
    ## we have overlap
    nowStr = u''.join(nowcodeList)
    len1 = len(sourcecodeList)
    match = False
    for i in range(0, len1, 1):
        compareStr = u''.join(sourcecodeList[i:])
        if nowStr.startswith(compareStr):
            resultList = sourcecodeList[0:i]
            resultList.extend(nowcodeList)
            match = True
            break
    if not match:
        resultList.extend(nowcodeList)
    print "Result len {}".format(len(resultList))
    return resultList

def generateMarkdown(fileHandle, title, url, desc, example, sourcecode):
    fileHandle.write(u"## {}\n".format(title).encode("UTF-8"))
    fileHandle.write(u"<{}>\n".format(url).encode("UTF-8"))
    fileHandle.write(u"### {}\n".format(desc).encode("UTF-8"))
    fileHandle.write(u"### {}\n".format(example).encode("UTF-8"))
    fileHandle.write("### Source code:" + os.linesep)
    fileHandle.write("```" + os.linesep)
    fileHandle.write(sourcecode.encode("UTF-8"))
    fileHandle.write(os.linesep)
    fileHandle.write("```" + os.linesep)
    return


f = open('algo_adv_now', 'wb')
NC_URL_BASE_ALGO = "http://www.lintcode.com/zh-cn/ladder/1/"
NC_URL_ADV_ALGO = "http://www.lintcode.com/zh-cn/ladder/4/"
NC_URL_SYSTEM_DESIGN = "http://www.lintcode.com/zh-cn/ladder/8/"
xpath_require_cache = ["//a[@href='#required14']",
                       "//a[@href='#required52']",
                       "//a[@href='#required15']",
                       "//a[@href='#required53']",
                       "//a[@href='#required54']",
                       "//a[@href='#required55']",
                       "//a[@href='#required56']"]

xpath_optional_cache = ["//a[@href='#optional14']",
                        "//a[@href='#optional52']",
                        "//a[@href='#optional15']",
                        "//a[@href='#optional53']",
                        "//a[@href='#optional54']",
                        "//a[@href='#optional55']",
                        "//a[@href='#optional56']"]
xpath_active_tab = []
if not OPTION:
    xpath_active_tab = ["div#required14.tab-pane.active",
                        "div#required52.tab-pane.active",
                        "div#required15.tab-pane.active",
                        "div#required53.tab-pane.active",
                        "div#required54.tab-pane.active",
                        "div#required55.tab-pane.active",
                        "div#required56.tab-pane.active"]
else:
    xpath_active_tab = ["div#optional14.tab-pane.active",
                        "div#optional52.tab-pane.active",
                        "div#optional15.tab-pane.active",
                        "div#optional53.tab-pane.active",
                        "div#optional54.tab-pane.active",
                        "div#optional55.tab-pane.active",
                        "div#optional56.tab-pane.active"]
index_tab = 0

CHROME_DRIVER_PATH = "/usr/local/bin/chromedriver"
os.environ["webdriver.chrome.driver"] = CHROME_DRIVER_PATH
driver = webdriver.Chrome(CHROME_DRIVER_PATH)
driver.get(NC_URL_ADV_ALGO)
time.sleep(5)
user = driver.find_element_by_name("username_or_email")
password = driver.find_element_by_id("inputPassword")
#submitButton = driver.find_element(By.CLASS_NAME, "btn btn-success btn-block")
submitButton = driver.find_element_by_css_selector(".btn.btn-success.btn-block")
user.send_keys("<your_user_name>@gmail.com")
password.send_keys("your_password")
submitButton.click()
time.sleep(5)

alreadyPayedButton = driver.find_element_by_css_selector("a.btn.btn-success")
alreadyPayedButton.click()
print "Sleeping 1 s..."
time.sleep(1)

## optinal
if OPTION:
    for opt in xpath_optional_cache:
        driver.find_element_by_xpath(opt).click()

mainWin = driver.current_window_handle
panels = driver.find_elements_by_css_selector("section.panel.m-b-sm")
driver.maximize_window()

for panel in panels:
    groupTitle = panel.find_element_by_class_name("text-md").text
    f.write(u"# {}".format(groupTitle).encode("UTF-8"))
    f.write(os.linesep)
    ## click must do
    #problemGroup = panel.find_element_by_id("problem_list_pagination")
    problemGroup = panel.find_element_by_css_selector(xpath_active_tab[index_tab])
    index_tab += 1
    problems = problemGroup.find_elements_by_css_selector("a.problem-panel.list-group-item")
    for problem in problems:
        problemUrl = problem.get_attribute("href") ###
        title = problem.find_element_by_css_selector("span.m-l-sm.title").text
        print u"num of problem is {}".format(len(problems))
        print u"title is {}".format(title)
        print u"problem URL is {}".format(problemUrl)
        print u"problem text is : {}".format(problem.text)
        problem.click()
        time.sleep(2)
        #WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "submit-btn")))
        if len(driver.window_handles) < 2:
            raise Exception('Failed to open problem Tab!')
        ## new problem webpage
        newWin = driver.window_handles[-1]
        driver.switch_to_window(newWin)
        driver.maximize_window()
        ## select C++ language
        driver.find_element_by_css_selector("select#code-language.form-control.m-t-sm.input-sm.m-r-xs.pull-left  > option[value='C++']").click()
        ## description
        desc = driver.find_element_by_class_name("panel-body").text ###
        ## 样列
        example = driver.find_element_by_css_selector("div.m-t-lg.m-b-lg").text

        ## code
        code = driver.find_element_by_css_selector("div.ace_layer.ace_text-layer")
        editor_text = driver.find_element_by_class_name("ace_text-input")
        #editor_text.click()
        sourcecodeList = code.text.splitlines()
        preStart, preEnd, preTotalNum = getLineRange(driver)


        ## get max page line
        for xx in range(0, 100):
            editor_text.send_keys(Keys.PAGE_DOWN)
        time.sleep(1)
        temp1, maxPage, temp3 = getLineRange(driver)
        print "maxPage in total = {}".format(maxPage)

        ## go to first line
        for l in range(0, 100):
            editor_text.send_keys(Keys.PAGE_UP)

        sourcecodeList = driver.find_element_by_css_selector("div.ace_layer.ace_text-layer").text.splitlines()
        nowStart, nowEnd, nowTotalNum = getLineRange(driver)

        for l in range(1, nowEnd):
            editor_text.send_keys(Keys.ARROW_DOWN)

        nowStart, nowEnd, nowTotalNum = getLineRange(driver)

        while nowEnd < maxPage:
            editor_text.send_keys(Keys.ARROW_DOWN)
            nowStart, nowEnd, nowTotalNum = getLineRange(driver)
            nowcodeList    = driver.find_element_by_css_selector("div.ace_layer.ace_text-layer").text.splitlines()
            if nowcodeList[-1] != sourcecodeList[-1]:
                sourcecodeList.append(nowcodeList[-1])

        finalSourceCodeText = '\n'.join(sourcecodeList)
        print finalSourceCodeText
        generateMarkdown(f, title, problemUrl, desc, example, finalSourceCodeText)
        driver.close()
        driver.switch_to_window(mainWin)
        driver.maximize_window()

        ######### TEST REOVE ME
        #problemCount = problemCount + 1
        #if problemCount > 2:
        #    driver.quit()
        #    f.close()
        #    exit()
        ##########

f.close()
driver.quit()
