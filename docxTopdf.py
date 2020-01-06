import sys
import os
import comtypes.client

wdFormatPDF = 17

in_file = os.path.abspath("C:\\Users\\sahat\\Desktop\\Machine Learning\\Regression\\Crunch Prep Reading Comprehension.doc")
out_file = os.path.abspath("C:\\Users\\sahat\\Desktop\\Machine Learning\\Regression\\cc.pdf")

word = comtypes.client.CreateObject('Word.Application')

doc = word.Documents.Open(in_file)
doc.SaveAs(out_file, FileFormat=wdFormatPDF)
doc.Close()
word.Quit()