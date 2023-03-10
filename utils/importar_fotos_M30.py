import selenium
from selenium.webdriver import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import keyboard
import xml.etree.ElementTree as ET

'''
Para obtener las imágenes de la M-30 del ayuntamiento. Se actualizan cada 10 min.
Las URL están en el arhivo camaras.xml. El programa lo recorre y con selenium va guardando las imágenes en la carpeta "Descargas".
Son 36 fotos y tarda unos 3.5 min
'''

def importar_fotos_M30():
    #time.sleep(500)
    camaras = ET.parse("C:/Users/mcalv/Desktop/Proyectos/Machine learning/camaras.xml")
    raiz_camaras = camaras.getroot()
    url_fotos = []

    for i in raiz_camaras.iter():
        if i.tag == "URL":
            url_fotos.append(i.text)
    len(url_fotos)

    for i in url_fotos:
        driver = selenium.webdriver.Chrome(executable_path=ChromeDriverManager().install())
        driver.get('http:'+i)
        driver.maximize_window()
        action = ActionChains(driver)
        # right click operation
        action.context_click(driver.find_element(By.XPATH,"/html/body/img"))
        action.perform()
        time.sleep(0.5)
        keyboard.press_and_release('DOWN')
        time.sleep(0.5)
        keyboard.press_and_release('DOWN')
        time.sleep(0.5)
        keyboard.press_and_release('ENTER')
        time.sleep(0.5)
        keyboard.press_and_release('ENTER')
        time.sleep(0.5)
        driver.close()


    