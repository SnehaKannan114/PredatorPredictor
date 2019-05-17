from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty

Builder.load_string("""
<Button>:
    font_size: 30
    color: 1, 1, 1, 1
    background_color: 0.3, 0.5 , 0.7, 1
    size_hint_y: None

<MainWindow>:
    name: "main"

    GridLayout:
        cols:1
        size: root.width - 300, root.height -300

        Button:
            size: 120, 120
            text:"Classify New Tweet"
            on_release: app.root.current = 'second'

        Button:
            size: 120, 120
            text:"Potential Abusers"
            on_release: app.root.current = 'third'
            on_release: app.third_window_var.populate_abuser_list()

        Button:
            size: 120, 120
            text:"Retrain Model"
            on_release: root.btn()
            on_release: root.retraining_model()

        Button:
            size: 120, 120
            text:"Check Accuracy"
            on_release: app.root.current = 'fifth'
            on_release: app.fifth_window_var.display_accuracy()
            
        Button:
            size: 120, 120
            text:"Exit"
            on_release: app.close()            

<SecondWindow>:
    name: "second"

    GridLayout:
        cols:1

        Label:
            size_hint_y: None
            text: "Enter text to be classified"
            font_size: 20
            height: 40

        TextInput:
            multiline:True
            id: fetch_raw_tweet
            
        Button:
            size: 50, 50
            text: "Classify"
            on_release: root.process_command() 

        TextInput:
            size_hint_y: None
            text: "The classified text is:  "
            id: processed_tweet
            font_size: 20
            height: 200
            
        Button:
            size: 50, 50
            text: "Go Back"
            on_release: app.root.current = "main"    

<ThirdWindow>:
    name: "third"

    GridLayout:
        cols:1

        Label:
            size_hint_y: None
            text: "Potential Abuser List "
            font_size: 20
            height: 40

        TextInput:
            id: abuser_list
            multiline:True
        
        Button:
            size: 50, 50
            text: "Go Back"
            on_release: app.root.current = "main"    

<FourthWindow>:
    name: "fourth"

    GridLayout:
        cols:1

        Label:
            text: "Pop up window "

        TextInput:
            multiline:False
            
        Button:
            size: 50, 50
            text: "Go Back"
            on_release: app.root.current = "main"    
            
<FifthWindow>:
    name: "fifth"
    GridLayout:
        cols:1

        Label:
            size_hint_y: None
            text: "Accuracy of the dataset is "
            font_size: 20
            height: 40

        TextInput:
            id: accuracy_disp
            multiline:True

        Button:
            size: 50, 50
            text: "Go Back"
            on_release: app.root.current = "main"       
            
<PopUpClass>:
    Label:
        text: "The model is being retrained. This might take a while!"
        size_hint: 0.6, 0.2
        pos_hint: {"x":0.2, "top":1}       

""")
# Declare all screens
class MainWindow(Screen):
        
    def btn(self): show_popup()

    def retraining_model(self): retrain_model()    

class SecondWindow(Screen):

    def process_command(self):
        # Text string from the user text input 
        user_input = self.ids.fetch_raw_tweet.text
        
        #call function to classify tweet here.
          
        # Displaying the output -> right now i'm just displaying the same tweet again
        self.ids.processed_tweet.text = 'The tweet: ' + user_input +' is classified as .... by the system'  
        #clearing user input in first input box
        self.ids.fetch_raw_tweet.text = ''
#=============================================================================
# for reference 

#          def command_dict(self):
#          return {
#             'one': self.command_one,
#             'two': self.command_two,
#             'three': self.command_three
#         }
# 
#     def process_command(self):
#         # We grab the text from the user text input as a key
#         command_key = self.ids.fetch_key_and_process_command.text
# 
#         # We then use that key in the command's built in 'get_method' because it is a dict
#         # then we store it into a variable for later use
#         called_command = self.command_dict().get(command_key, 'default')
# 
#         try:
#             # The variable is a method, so by adding we can call it by simple adding your typical () to the end of it.
#             called_command()
# 
#         except TypeError:
#             # However we use an exception clause to catch in case people enter a key that doesn't exist
#             self.ids.fetch_key_and_process_command.text = 'Sorry, there is no command key: ' + command_key
# 
#     # These are the three commands we call from our command dict.
#     def command_one(self):
#         self.ids.fetch_key_and_process_command.text = 'Command One has Been Processed'
# 
#     def command_two(self):
#         self.ids.fetch_key_and_process_command.text = 'Command Two has Been Processed'
# 
#     def command_three(self):
#         self.ids.fetch_key_and_process_command.text = 'Command Three has been Processed'
# =============================================================================
      
class ThirdWindow(Screen):
    def populate_abuser_list(self):
    #populate the textbox with names of potential abusers using the id: abuser_list
    #right now i'm just printing  some text
        self.ids.abuser_list.text = 'ABCDEFIJKLMNOPQURISTUVWXYZ'

class FourthWindow(Screen):
    pass

class FifthWindow(Screen):
    def display_accuracy(self):
    #display the accuracy in the textbox using the id: accuracy_disp
    #right now i'm just printing  some text
        self.ids.accuracy_disp.text = 'Sneha is a dumbass'

class PopUpClass(FloatLayout):
    pass

def show_popup():
    show = PopUpClass()
    popupWindow = Popup(title="Popup Window", content=show, size_hint=(None,None),size=(400,400))
    popupWindow.open()
    
def retrain_model():
#add code to retrain model here
    #print statement for testing
    print('retraining model')    

# Screen manager
sm = ScreenManager()
sm.add_widget(MainWindow(name='main'))
sm.add_widget(SecondWindow(name='second'))
sm.add_widget(ThirdWindow(name='third'))
sm.add_widget(FourthWindow(name='fourth'))
sm.add_widget(FifthWindow(name='fifth'))

class TestApp(App):
    third_window_var = sm.get_screen('third')
    fifth_window_var = sm.get_screen('fifth')
    
    def build(self):
        return sm

if __name__ == '__main__':
    TestApp().run()