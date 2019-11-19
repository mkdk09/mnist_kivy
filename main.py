# coding: UTF-8
from kivy.app import App
from kivy.config import Config

# Config関係は他のモジュールがインポートされる前に行う
Config.set('graphics', 'width', '300')
Config.set('graphics', 'height', '340')
Config.write()

from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.utils import get_color_from_hex
from kivy.core.window import Window
from kivy.properties import StringProperty

from PIL import Image
import numpy as np
# import learning
from keras.models import load_model

class MyPaintWidget(Widget):
    line_width = 10 # 線の太さ
    color = get_color_from_hex('#ffffff') # 線の色

    def on_touch_down(self, touch):
        if Widget.on_touch_down(self, touch):
            return

        with self.canvas:
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=self.line_width)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]

    def set_color(self):
        self.canvas.add(Color(*self.color))

class MyCanvasWidget(Widget):

    def clear_canvas(self):
        MyPaintWidget.clear_canvas(self)

class MyPaintApp(App):
    result = StringProperty()

    def __init__(self, **kwargs):
        super(MyPaintApp, self).__init__(**kwargs)
        self.title = '手書き数字認識テスト'
        self.result = ''

        self.model = load_model('./mnist_cnn_model.h5')

    def build(self):
        self.painter = MyCanvasWidget()
        # 起動時の色の設定を行う
        self.painter.ids['paint_area'].set_color()
        return self.painter

    def clear_canvas(self):
        self.painter.ids['paint_area'].canvas.clear()
        self.painter.ids['paint_area'].set_color() # クリアした後に再び色をセット
        self.result = ''

    def predict(self):
        self.painter.export_to_png('canvas.png')

        # image = Image.open('canvas.png').crop((0, 0, 600, 600)).convert('L')
        image = Image.open('canvas.png').crop((0, 0, 300, 280)).convert('L')
        # image = Image.open('canvas.png').convert('L')
        image.save('./transfer.png')
        image = image.resize((28, 28))
        image = np.array(image)
        image = image.reshape([1,28,28,1])
        # print(image)
        ans = self.model.predict(image)
        # print(ans)
        self.result = str(np.argmax(ans))
        # print('This Digit is ...', np.argmax(ans))

if __name__ == '__main__':
    Window.clearcolor = get_color_from_hex('#000000') # ウィンドウの色を黒色に変更する
    MyPaintApp().run()