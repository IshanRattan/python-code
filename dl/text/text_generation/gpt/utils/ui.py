
from functools import lru_cache
from configs import config

class JS():

    @classmethod
    @lru_cache(maxsize=4)
    def recordAudio(cls):
        return """function audioRecord() {var xPathRes = document.evaluate ('//*[@id="audio"]//button', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
            xPathRes.singleNodeValue.click();

            let btn = document.getElementById('btn');

            if (btn.innerText.includes('Stop')) {
                btn.style.backgroundColor = '#E0115F';}
            else {
                btn.style.backgroundColor = '';};}"""

    @classmethod
    @lru_cache(maxsize=4)
    def validateState(cls):
        return """function throwException() {let element = document.getElementById('audio').innerText;
                if (element.includes('Stop')) {
                    throw Error('Recording...');};}"""

    @classmethod
    def appTitle(cls):
        return f"<h1 style='text-align: center; margin-bottom: 1rem'>{config.appTitle}</h1>"