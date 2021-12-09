####BO MON KY THUAT PHAN MEM-NTU####
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import os
import csv
import json
import requests
import argparse
import re
import datetime
import unidecode
import concurrent.futures
from pathlib import Path 
import webbrowser



CONNECTIONS = 100
class FBScraper():
    def __init__(self, page, output, token, since, until, folder=None):
        self.token = token
        self.output = output
        self.since = since
        self.until = until
        folder_param = ('&folder=' + folder) if folder is not None else ''
        self.uri = self.build_url('{}/conversations?fields=participants,link&limit=400{}', page, folder_param)

    def build_url(self, endpoint, *params):
        buildres = "https://graph.facebook.com/v3.1/" + endpoint.format(*params) + '&access_token={}'.format(self.token)
        print("URL: ", buildres)
        return buildres

    def scrape_thread(self, url, lst):
        if self.since:
            matches = re.findall('&until=(\d+)', url)
            if matches and int(matches[0]) <= self.since:
                return lst

        messages = requests.get(url).json()
        for m in messages['data']:
            time = datetime.datetime.strptime(m['created_time'], '%Y-%m-%dT%H:%M:%S+0000').replace(tzinfo=datetime.timezone.utc).timestamp()
            #TRIEU: compare here
            if self.since and time < self.since:
                continue
            if self.until and time > self.until:
                continue
            lst.append({
                'time': m['created_time'].replace('+0000', '').replace('T', ' '),
                'message': m['message'],
                'attachments': m.get('attachments', {}).get('data', [{}])[0].get('image_data', {}).get('url', ''),
                'shares': m.get('shares', {}).get('data', [{}])[0].get('name', ''),
                'from_id': m['from']['id']
            })
        # if messages['data']:
        #     print(' +', len(messages['data']))
        next = messages.get('paging', {}).get('next', '')
        if next:
            self.scrape_thread(next, lst)
        return lst

    def get_messages(self, t):
        extra_params = (('&since=' + str(self.since)) if self.since else '') + (('&until=' + str(self.until)) if self.until else '')
        url = self.build_url('{}/messages?fields=from,created_time,message,shares,attachments&limit=400' + extra_params, t['id'])
        thread = self.scrape_thread(url, [])
        if thread:
            print(
                thread[0]['time'], 
                t['id'].ljust(20), 
                str(len(thread)).rjust(3) + ' from', 
                unidecode.unidecode(t['participants']['data'][0]['name'])
            )            
            id_map = {p['id']: p['name'] for p in t['participants']['data']}
            for message in thread:
                message['from'] = id_map[message['from_id']]

            return [{
                # 'page_id': t['participants']['data'][1]['id'],
                # 'page_name': t['participants']['data'][1]['name'],
                # 'user_id': t['participants']['data'][0]['id'],
                # 'user_name': t['participants']['data'][0]['name'],
                'url': t['link'],
            }] + list(reversed(thread))
        return []
        
    def scrape_thread_list(self, threads, count):
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS) as executor:
            futures = (executor.submit(self.get_messages, conv) for conv in threads['data'])
            for future in concurrent.futures.as_completed(futures):
                messages = future.result()
                for message in messages:
                    self.writer.writerow(message)
        next = threads.get('paging', {}).get('next', '')
        if next and count > 1:
            self.scrape_thread_list(requests.get(next).json(), count - 1)
        

    def run(self):
        output = open(self.output, 'w', newline="\n", encoding="utf-8")
        threads = requests.get(self.uri).json()
        if 'error' in threads:
            print(threads)
            return

        fieldnames = ['from_id', 'from', 'time', 'message', 'attachments', 'shares', 'url']
        self.writer = csv.DictWriter(output, dialect='excel', fieldnames=fieldnames, extrasaction='ignore', quoting=csv.QUOTE_NONNUMERIC)
        self.writer.writerow(dict((n, n) for n in fieldnames))
        self.scrape_thread_list(threads, 20)
        output.close()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(468, 341)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_from = QtWidgets.QLabel(self.centralwidget)
        self.label_from.setText("From")
        self.label_from.setObjectName("label_from")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_from)
        self.fromdate = QtWidgets.QDateTimeEdit(self.centralwidget)
        self.fromdate.setDateTime(QtCore.QDateTime(QtCore.QDate(2020, 1, 1), QtCore.QTime(0, 0, 0)))
        self.fromdate.setDisplayFormat("dd/MM/yyyy HH:mm")
        self.fromdate.setCalendarPopup(True)
        self.fromdate.setObjectName("fromdate")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.fromdate)
        self.label_to = QtWidgets.QLabel(self.centralwidget)
        self.label_to.setText("To")
        self.label_to.setObjectName("label_to")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_to)
        self.todate = QtWidgets.QDateTimeEdit(self.centralwidget)
        self.todate.setDateTime(QtCore.QDateTime(QtCore.QDate(2021, 1, 1), QtCore.QTime(0, 0, 0)))
        self.todate.setDisplayFormat("dd/MM/yyyy HH:mm")
        self.todate.setCalendarPopup(True)
        self.todate.setObjectName("todate")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.todate)
        self.pageid = QtWidgets.QLabel(self.centralwidget)
        self.pageid.setMaximumSize(QtCore.QSize(16777215, 31))
        self.pageid.setText("Page ID")
        self.pageid.setObjectName("pageid")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.pageid)
        self.pageid_text = QtWidgets.QTextEdit(self.centralwidget)
        self.pageid_text.setMaximumSize(QtCore.QSize(16777215, 21))
        self.pageid_text.setObjectName("pageid_text")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.pageid_text)
        self.token = QtWidgets.QLabel(self.centralwidget)
        self.token.setMaximumSize(QtCore.QSize(16777215, 31))
        self.token.setText("Token")
        self.token.setObjectName("token")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.token)
        self.token_text = QtWidgets.QTextEdit(self.centralwidget)
        self.token_text.setMaximumSize(QtCore.QSize(16777215, 21))
        self.token_text.setObjectName("token_text")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.token_text)
        self.btn1 = QtWidgets.QPushButton(self.centralwidget)
        self.btn1.setMaximumSize(QtCore.QSize(89, 25))
        self.btn1.setObjectName("btn1")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.btn1)
        self.listView = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.listView.setMaximumSize(QtCore.QSize(16777215, 181))
        self.listView.setObjectName("listView")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.listView)        
        self.horizontalLayout.addLayout(self.formLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 468, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuAbout = QtWidgets.QMenu(self.menubar)
        self.menuAbout.setObjectName("menuAbout")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        ####action Quit
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.menuFile.addAction(self.actionQuit)
        ####action About 
        self.actionAbout=QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.menuAbout.addAction(self.actionAbout)
        ####action actionHelp_Contact
        self.actionHelp_Contact=QtWidgets.QAction(MainWindow)
        self.actionHelp_Contact.setObjectName("actionHelp_Contact")
        self.menuHelp.addAction(self.actionHelp_Contact)
        
        self.actionHelp_Guide=QtWidgets.QAction(MainWindow)
        self.actionHelp_Guide.setObjectName("actionHelp_Guide")
        self.menuHelp.addAction(self.actionHelp_Guide)
        ####

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "NTU_Crawler"))
        self.btn1.setText(_translate("MainWindow", "Get DATA"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.menuAbout.setTitle(_translate("MainWindow", "About"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
        self.actionQuit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        ##Infor
        self.actionAbout.setText(_translate("MainWindow", "Information"))
        self.actionAbout.setShortcut(_translate("MainWindow", "Ctrl+I"))
        ##Contact
        self.actionHelp_Contact.setText(_translate("MainWindow", "Contact"))
        ##Guide
        self.actionHelp_Guide.setText(_translate("MainWindow", "Guide"))
        self.actionHelp_Guide.setShortcut(_translate("MainWindow", "Ctrl+H"))



class crawler(QMainWindow,Ui_MainWindow,FBScraper):
    def __init__(self):
        super(crawler,self).__init__()
        self.setupUi(self)
        self.actionQuit.triggered.connect(self.close)
        self.actionAbout.triggered.connect(self.showAbout)
        self.actionHelp_Contact.triggered.connect(self.showContact)
        self.actionHelp_Guide.triggered.connect(self.showGuide)
        # self.listView.
        self.listView.insertPlainText("1. Setup params\n")
        self.listView.insertPlainText("-->Look for these parameters at the following URL: \n")
        self.listView.insertPlainText("https://developers.facebook.com/tools/explorer\n")
        self.btn1.clicked.connect(self.processing)
        # self.btn1.clicked.connect(self.click_getDATA)   
    def showAbout(self):
        QMessageBox.about(self, "Information", "This tool is supported by Nha Trang University")
    def showContact(self):
        QMessageBox.about(self, "Contact", "Software Engineering Department,\nFaculty of Information Technology,\nNha Trang University")
    def showGuide(self):
        cwd = os.path.dirname(os.path.realpath(__file__))
        file_path=cwd+"/guide/guide.pdf"
        webbrowser.open_new(file_path)
    def processing(self):
        ## processing
        self.listView.insertPlainText("2. Processing to get data->>>>>>>>\n")
        self.listView.insertPlainText(".........................Waiting.........................\n")
        self.click_getDATA()
    def click_getDATA(self):  
        ##savefile
        output, name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File',"*.csv")      
        ##fromdate
        var_fromdatetime=self.fromdate.dateTime()
        from_datetime_object=var_fromdatetime.toPyDateTime()
        from_timestamp = datetime.datetime.timestamp(from_datetime_object)
        since=int(from_timestamp)
        ##todate
        var_todatetime=self.todate.dateTime()
        to_datetime_object=var_todatetime.toPyDateTime()
        to_timestamp = datetime.datetime.timestamp(to_datetime_object)
        until=int(to_timestamp)
        ##pageid
        page=int(self.pageid_text.toPlainText()) 
        ##token
        token=self.token_text.toPlainText()
        folder=None
        FBScraper(page,output,token, since, until, folder=None).run() 
        self.listView.insertPlainText("-------------------->DONE<-------------------\n")
        self.listView.insertPlainText("3. Your file has been saved in:\n"+output)

if __name__=='__main__':
    import sys
    app = QApplication(sys.argv)
    window = crawler()
    window.show() # IMPORTANT!!!!! Windows are hidden by default.
    # Start the event loop.
    app.exec_()