from telegram.ext import Updater, CommandHandler, MessageHandler  # Подключение нужных функций из модуля telegram
from telegram.ext import Filters as F
from telegram.constants import PARSEMODE_MARKDOWN
import random
from telegram import InlineKeyboardButton, ReplyKeyboardMarkup
from model.base_model_train import load_data
from sklearn.neural_network import MLPClassifier
from model.base_model_train import extract_feature
from pydub import AudioSegment


def infer(model, path):
    sound = AudioSegment.from_file(path, 'ogg')
    new_audioname = path[:-4]+'.wav'
    sound.export(new_audioname, format="wav")
    X = extract_feature(new_audioname, True, True, True)
    return model.predict(X.reshape(1, -1))[0]


def start (update, context):
    name = update.message.from_user.first_name
    context.bot.send_message(chat_id=update.message.chat_id, text='''
Hello! 
This bot can recognize 4 human emotions by your speech. 
Wanna try?
*Just send me something by voicemail!*
    ''', parse_mode=PARSEMODE_MARKDOWN)


def audio(update, context):
    name = update.message.from_user.first_name
    user_id = update.message.chat_id
    print(name, user_id)

    audio = context.bot.getFile(update.message.voice.file_id)
    audioname = 'audio_{}_{}.ogg'.format(random.randint(0, 100), user_id)
    audio.download(audioname)
    ret = infer(model, audioname)
    msg = '''
Seems like... Emotion of your voice message was {}
    '''.format(ret)
    context.bot.send_message(chat_id=update.message.chat_id, text=msg)


TOKEN = 0###### Здесь должен быть ваш токен


updater = Updater(token=TOKEN, workers=4)  # Создаём переменную типа (Updater) с аргументом token, который равен переменной TOKEN

dispatcher = updater.dispatcher  # Создаём диспетчер, который будет хранить все Handler'ы.

dispatcher.add_handler(CommandHandler('start', start, run_async=True))

dispatcher.add_handler(MessageHandler(F.voice, audio, run_async=True))

updater.start_polling()
updater.idle()
