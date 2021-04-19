import wx
import wx.xrc
import pandas as pd
from keras.models import load_model
from sklearn import preprocessing
import time

# Adding champs list
herolist = []
file_dir = 'D:/Project/ASD Project/input/'
champs = pd.read_csv(file_dir+'champs.csv')
list = champs.values.tolist()
for i in range(len(list)):
    herolist.append(list[i][0])

# load processed data
data = pd.read_csv('D:/Project/ASD Project2/DataFrame.csv', index_col=0)
X_t = data[data.columns.difference(['T1 win'])]

# loading model
model = load_model('D:/Project/ASD Project2/Neural Network Model.h5')


def cal_T1_win_rate(Blue_CARRY, Blue_SUPPORT, Blue_JUNGLE, Blue_MID, Blue_TOP,
                 Red_CARRY, Red_SUPPORT, Red_JUNGLE, Red_MID, Red_TOP):
    print("start")
    s = pd.Series([Blue_CARRY, Blue_SUPPORT, Blue_JUNGLE, Blue_MID, Blue_TOP,
                   Red_CARRY, Red_SUPPORT, Red_JUNGLE, Red_MID, Red_TOP,
                   'NA1', 8], index=data[data.columns.difference(['T1 win'])].columns, name='s')
    X = X_t.copy(deep=True)
    X = X.append(s)
    le = preprocessing.LabelEncoder()
    le_t = X.apply(le.fit)
    X_1 = X.apply(le.fit_transform)
    enc = preprocessing.OneHotEncoder()
    enc_t = enc.fit(X_1)
    X_2 = enc_t.transform(X_1)

    # predicting win rate
    T1_win_rate = model.predict(X_2[-1])[0][1]
    return float(T1_win_rate)

class MyPanel7(wx.Panel):

    def __init__(self, parent, id):
        # wx.Panel.__init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition
        # , size=wx.Size(500, 400),style=wx.TAB_TRAVERSAL)
        wx.Panel.__init__(self, parent, id)
        try:
            image_file = 'background.bmp'
            to_bmp_image = wx.Image(image_file, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            self.bitmap = wx.StaticBitmap(self, -1, to_bmp_image, (0, 0))
            image_width = to_bmp_image.GetWidth()
            image_height = to_bmp_image.GetHeight()
            set_title = '%s %d x %d' % (image_file, to_bmp_image.GetWidth(), to_bmp_image.GetHeight())
            # parent.SetTitle(set_title)
        except IOError:
            print
            'Image file %s not found' % image_file
            raise SystemExit

        fgSizer1 = wx.FlexGridSizer(0, 3, 10, 40)
        fgSizer1.SetFlexibleDirection(wx.BOTH)
        fgSizer1.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        self.m_staticText1 = wx.StaticText(self, wx.ID_ANY, u"BLUE TEAM", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText1.Wrap(-1)
        self.m_staticText1.SetFont(wx.Font(14, 75, 90, 92, False, "Unispace"))
        self.m_staticText1.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_ACTIVECAPTION))
        fgSizer1.Add(self.m_staticText1, 0, wx.ALL, 5)

        self.m_staticText2 = wx.StaticText(self, wx.ID_ANY, u"HERO POSITION", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText2.Wrap(-1)
        self.m_staticText2.SetFont(wx.Font(14, 75, 90, 92, False, "Unispace"))
        self.m_staticText2.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_ACTIVECAPTION))
        fgSizer1.Add(self.m_staticText2, 0,  wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5)

        self.m_staticText3 = wx.StaticText(self, wx.ID_ANY, u"RED TEAM", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText3.Wrap(-1)
        self.m_staticText3.SetFont(wx.Font(14, 75, 90, 92, False, "Unispace"))
        self.m_staticText3.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_ACTIVECAPTION))
        fgSizer1.Add(self.m_staticText3, 0, wx.ALL | wx.ALIGN_RIGHT, 5)

        m_comboBox1Choices = herolist
        self.m_comboBox1 = wx.ComboBox(self.bitmap, -1, u"Select a hero!", wx.DefaultPosition, wx.DefaultSize,
                                       m_comboBox1Choices, wx.CB_READONLY | wx.CB_SORT)
        self.m_comboBox1.SetFont(wx.Font(10, 75, 90, 92, False, "Unispace"))
        self.m_comboBox1.SetMinSize(wx.Size(110, -1))

        fgSizer1.Add(self.m_comboBox1, 0, wx.ALL, 5)

        self.m_staticText4 = wx.StaticText(self, wx.ID_ANY, u"TOP", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText4.Wrap(-1)
        self.m_staticText4.SetFont(wx.Font(14, 75, 90, 92, False, "Unispace"))
        self.m_staticText4.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_ACTIVECAPTION))
        fgSizer1.Add(self.m_staticText4, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 5)

        m_comboBox2Choices = herolist
        self.m_comboBox2 = wx.ComboBox(self.bitmap, -1, u"Select a hero!", wx.DefaultPosition, wx.DefaultSize,
                                       m_comboBox2Choices, wx.CB_READONLY | wx.CB_SORT)
        self.m_comboBox2.SetFont(wx.Font(10, 75, 90, 92, False, "Unispace"))
        self.m_comboBox2.SetMinSize(wx.Size(110, -1))

        fgSizer1.Add(self.m_comboBox2, 0, wx.ALL, 5)

        m_comboBox3Choices = herolist
        self.m_comboBox3 = wx.ComboBox(self.bitmap, -1, u"Select a hero!", wx.DefaultPosition, wx.DefaultSize,
                                       m_comboBox3Choices, wx.CB_READONLY | wx.CB_SORT)
        self.m_comboBox3.SetFont(wx.Font(10, 75, 90, 92, False, "Unispace"))
        self.m_comboBox3.SetMinSize(wx.Size(110, -1))

        fgSizer1.Add(self.m_comboBox3, 0, wx.ALL, 5)

        self.m_staticText5 = wx.StaticText(self, wx.ID_ANY, u"JUNGLE", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText5.Wrap(-1)
        self.m_staticText5.SetFont(wx.Font(14, 75, 90, 92, False, "Unispace"))
        self.m_staticText5.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_ACTIVECAPTION))
        fgSizer1.Add(self.m_staticText5, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        m_comboBox4Choices = herolist
        self.m_comboBox4 = wx.ComboBox(self.bitmap, -1, u"Select a hero!", wx.DefaultPosition, wx.DefaultSize,
                                       m_comboBox4Choices, wx.CB_READONLY | wx.CB_SORT)
        self.m_comboBox4.SetFont(wx.Font(10, 75, 90, 92, False, "Unispace"))
        self.m_comboBox4.SetMinSize(wx.Size(110, -1))

        fgSizer1.Add(self.m_comboBox4, 0, wx.ALL, 5)

        m_comboBox5Choices = herolist
        self.m_comboBox5 = wx.ComboBox(self.bitmap, -1, u"Select a hero!", wx.DefaultPosition, wx.DefaultSize,
                                       m_comboBox5Choices, wx.CB_READONLY | wx.CB_SORT)
        self.m_comboBox5.SetFont(wx.Font(10, 75, 90, 92, False, "Unispace"))
        self.m_comboBox5.SetMinSize(wx.Size(110, -1))

        fgSizer1.Add(self.m_comboBox5, 0, wx.ALL, 5)

        self.m_staticText6 = wx.StaticText(self, wx.ID_ANY, u"MID", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText6.Wrap(-1)
        self.m_staticText6.SetFont(wx.Font(14, 75, 90, 92, False, "Unispace"))
        self.m_staticText6.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_ACTIVECAPTION))

        fgSizer1.Add(self.m_staticText6, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        m_comboBox6Choices = herolist
        self.m_comboBox6 = wx.ComboBox(self.bitmap, -1, u"Select a hero!", wx.DefaultPosition, wx.DefaultSize,
                                       m_comboBox6Choices, wx.CB_READONLY | wx.CB_SORT)
        self.m_comboBox6.SetFont(wx.Font(10, 75, 90, 92, False, "Unispace"))
        self.m_comboBox6.SetMinSize(wx.Size(110, -1))

        fgSizer1.Add(self.m_comboBox6, 0, wx.ALL, 5)

        m_comboBox8Choices = herolist
        self.m_comboBox8 = wx.ComboBox(self.bitmap, -1, u"Select a hero!", wx.DefaultPosition, wx.DefaultSize,
                                       m_comboBox8Choices, wx.CB_READONLY | wx.CB_SORT)
        self.m_comboBox8.SetFont(wx.Font(10, 75, 90, 92, False, "Unispace"))
        self.m_comboBox8.SetMinSize(wx.Size(110, -1))

        fgSizer1.Add(self.m_comboBox8, 0, wx.ALL, 5)

        self.m_staticText8 = wx.StaticText(self, wx.ID_ANY, u"CARRIER", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText8.Wrap(-1)
        self.m_staticText8.SetFont(wx.Font(14, 75, 90, 92, False, "Unispace"))
        self.m_staticText8.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_ACTIVECAPTION))

        fgSizer1.Add(self.m_staticText8, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        m_comboBox9Choices = herolist
        self.m_comboBox9 = wx.ComboBox(self.bitmap, -1, u"Select a hero!", wx.DefaultPosition, wx.DefaultSize,
                                       m_comboBox9Choices, wx.CB_READONLY | wx.CB_SORT)
        self.m_comboBox9.SetFont(wx.Font(10, 75, 90, 92, False, "Unispace"))
        self.m_comboBox9.SetMinSize(wx.Size(110, -1))

        fgSizer1.Add(self.m_comboBox9, 0, wx.ALL, 5)

        m_comboBox10Choices = herolist
        self.m_comboBox10 = wx.ComboBox(self.bitmap, -1, u"Select a hero!", wx.DefaultPosition, wx.DefaultSize,
                                        m_comboBox10Choices, wx.CB_READONLY | wx.CB_SORT)
        self.m_comboBox10.SetFont(wx.Font(10, 75, 90, 92, False, "Unispace"))
        self.m_comboBox10.SetMinSize(wx.Size(110, -1))

        fgSizer1.Add(self.m_comboBox10, 0, wx.ALL, 5)

        self.m_staticText9 = wx.StaticText(self, wx.ID_ANY, u"SUPPORT", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText9.Wrap(-1)
        self.m_staticText9.SetFont(wx.Font(14, 75, 90, 92, False, "Unispace"))
        self.m_staticText9.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_ACTIVECAPTION))

        fgSizer1.Add(self.m_staticText9, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        m_comboBox11Choices = herolist
        self.m_comboBox11 = wx.ComboBox(self.bitmap, -1, u"Select a hero!", wx.DefaultPosition, wx.DefaultSize,
                                        m_comboBox11Choices, wx.CB_READONLY | wx.CB_SORT)
        self.m_comboBox11.SetFont(wx.Font(10, 75, 90, 92, False, "Unispace"))
        self.m_comboBox11.SetMinSize(wx.Size(110, -1))

        fgSizer1.Add(self.m_comboBox11, 0, wx.ALL, 5)

        fgSizer1.AddSpacer(0)

        self.m_button1 = wx.Button( self.bitmap, wx.ID_ANY, u"CALCULATE", wx.DefaultPosition, (150, 40), 0)
        self.m_button1.SetFont( wx.Font( 14, 71, 90, 92, False, "Stencil" ) )
        self.m_button1.SetForegroundColour( wx.Colour( 255, 128, 0 ) )
        self.m_button1.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_HIGHLIGHT))
        fgSizer1.Add(self.m_button1, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)
        # self.button = wx.Button(self.bitmap, -1, label='Test', pos=(10, 10))

        fgSizer1.AddSpacer(0)

        self.m_textCtrl1 = wx.TextCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize,
                                       wx.TE_READONLY)
        self.m_textCtrl1.SetFont(wx.Font(16, 72, 90, 90, False, "Goudy Stout"))
        self.m_textCtrl1.SetMinSize(wx.Size(100, 40))

        fgSizer1.Add(self.m_textCtrl1, 0, wx.ALL|wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)

        self.m_staticText91 = wx.StaticText(self, wx.ID_ANY, u"WIN-RATE", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText91.Wrap(-1)
        self.m_staticText91.SetFont(wx.Font(14, 75, 90, 92, False, "Unispace"))
        self.m_staticText91.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_ACTIVECAPTION))

        fgSizer1.Add(self.m_staticText91, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        self.m_textCtrl2 = wx.TextCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize,
                                       wx.TE_READONLY)
        self.m_textCtrl2.SetFont(wx.Font(16, 72, 90, 90, False, "Goudy Stout"))
        self.m_textCtrl2.SetMinSize(wx.Size(100, 40))

        fgSizer1.Add(self.m_textCtrl2, 0, wx.ALL|wx.ALIGN_LEFT|wx.ALIGN_CENTER_VERTICAL, 5)

        self.SetSizer(fgSizer1)
        self.Layout()
        self.m_button1.Bind(wx.EVT_LEFT_DCLICK, self.calculate)

    def __del__(self):
        pass

    # calculate win rate

    def calculate(self, event):
        selectedhero=[]
        blue_top = self.m_comboBox1.GetValue()
        blue_jungle = self.m_comboBox3.GetValue()
        blue_mid = self.m_comboBox5.GetValue()
        blue_carrier = self.m_comboBox8.GetValue()
        blue_support = self.m_comboBox10.GetValue()
        red_top = self.m_comboBox2.GetValue()
        red_jungle = self.m_comboBox4.GetValue()
        red_mid = self.m_comboBox6.GetValue()
        red_carrier = self.m_comboBox9.GetValue()
        red_support = self.m_comboBox11.GetValue()
        T1_win_rate = round(cal_T1_win_rate(blue_carrier, blue_support, blue_jungle, blue_mid, blue_top,
                                      red_carrier, red_support, red_jungle, red_mid, red_top),4)
        T2_win_rate = round(1.0 - T1_win_rate, 4)
        T1 = T1_win_rate * 100
        T2 = T2_win_rate * 100
        self.m_textCtrl1.SetValue("%.2f%%" % T1)
        self.m_textCtrl2.SetValue("%.2f%%" % T2)


if __name__ == '__main__':
    # 做出窗口
    app = wx.PySimpleApp()
    frame = wx.Frame(None, -1, 'Win rate calculator', size=(500, 435))
    my_panel = MyPanel7(frame, -1)
    frame.Show()
    app.MainLoop()