import pyreadstat

qual_data = pyreadstat.read_sav(r"C:\Users\Mikkel\Documents\UNI\masters\SmartHome\SmartHomeSurvey\Dataset_Smart Home Experience.sav")

data = qual_data[0]
data["v_29"].values