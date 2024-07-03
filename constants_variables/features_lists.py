LAB_VARIABLES = {50816: 'OXYGEN', 50868: 'ANION GAP', 50825: 'TEMPERATURE', 50814: 'METHEMOGLOBIN',
                 51006: 'UREA NITROGEN', 50862: 'ALBUMIN', 51275: 'PTT', 51009: 'VANCOMYCIN',
                 51265: 'PLATELET COUNT', 50970: 'PHOSPHATE', 51301: 'WHITE BLOOD CELLS', 51300: 'WBC COUNT',
                 51279: 'RED BLOOD CELLS', 50817: 'OXYGEN SATURATION', 50802: "BASE EXCESS",
                 50912: 'CREATININE', 50813: 'LACTATE', 50867: "AMYLASE", 50956: "LIPASE", 51144: "BANDS",
                 50976: "PROTEIN, TOTAL", 50924: "FERRITIN", 50998: "TRANSFERRIN", 50821: "PO2",
                 50815: "O2 FLOW", 50818: "PCO2", 51248: "MCH", 51249: "MCHC", 51250: "MCV", 51237: "INR(PT)",
                 50885: "BILIRUBIN", 50960: "MAGNESIUM", 51002: "TROPONIN I", 51003: "TROPONIN T",
                 50804: "CALCULATED TOTAL CO2", 50893: 'CALCIUM, TOTAL', 50823: "REQUIRED O2",
                 50805: "CARBOXYHEMOGLOBIN", 51498: "SPECIFIC GRAVITY", 51484: "KETONE", 51514: "UROBILINOGEN",
                 51196: "D-DIMER", 51214: "FIBRINOGEN", 51297: "THROMBIN", 50971: 'POTASSIUM',
                 50822: 'POTASSIUM, WHOLE BLOOD', 50983: 'SODIUM', 51100: 'SODIUM URINE', 50824: "SODIUM, WHOLE BLOOD",
                 50861: "ALANINE AMINOTRANSFERASE (ALT)", 50878: "ASPARATE AMINOTRANSFERASE (AST)", 51221: 'HEMATOCRIT',
                 50810: 'HEMATOCRIT, CALCULATED', 51078: "CHLORIDE, URINE", 50902: "CHLORIDE",
                 50806: "CHLORIDE, WHOLE BLOOD", 50882: 'BICARBONATE', 50803: "CALCULATED BICARBONATE, WHOLE BLOOD",
                 51218: "GRANULOCYTE COUNT", 51255: "MYELOCYTES", 51256: "NEUTROPHILS",
                 51143: "ATYPICAL LYMPHOCYTES", 51251: "METAMYELOCYTES", 51146: 'BASOPHILS', 51122: "NUCLEATED RBC",
                 51257: "NUCLEATED RED CELLS", 51476: "EPITHELIAL CELLS", 51116: "LYMPHOCYTES",
                 51244: "LYMPHOCYTES", 51375: "LYMPHOCYTES", 51427: "LYMPHOCYTES", 51446: "LYMPHOCYTES",
                 51254: "MONOCYTES", 51120: "MONOCYTES", 51355: "MONOCYTES", 51379: "MONOCYTES",
                 51200: "EOSINOPHILS", 51114: "EOSINOPHILS", 51200: "EOSINOPHILS", 51347: "EOSINOPHILS",
                 51368: "EOSINOPHILS", 51419: "EOSINOPHILS", 51444: "EOSINOPHILS", 51126: "PROMYELOCYTES",
                 51269: "PROMYELOCYTES", 51361: "PROMYELOCYTES", 51437: "PROMYELOCYTES", 51456: "PROMYELOCYTES",
                 51283: "RETICULOCYTE COUNT, AUTOMATED", 51284: "RETICULOCYTE COUNT, MANUAL",
                 50809: "GLUCOSE", 50931: "GLUCOSE", 51478: "GLUCOSE", 50811: "HEMOGLOBIN",
                 51222: "HEMOGLOBIN-CHEMISTRY", 50820: "PH", 50831: "PH", 51094: "PH", 51491: "PH"}
CHART_VARIABLES = {  # -- GCS
    723: 'GCSVerbal', 454: 'GCSMotor', 184: 'GCSEyes',
    223900: "Verbal Response", 223901: "Motor Response", 220739: "Eye Opening",
    # -- HEART RATE
    211: "Heart Rate", 220045: "Heart Rate",
    # -- TEMPERATURE BODY
    223761: "Temperature Fahrenheit", 678: "Temperature F", 676: "TempC", 223762: "TempC",
    3655: "Temp Skin [C]", 677: "Temperature C (calc)", 3654: "Temp Rectal [F]", 679: "Temperature F (calc)",
    # -- FiO2
    3420: "FiO2", 190: "FiO2 set", 223835: "Inspired O2 Fraction (FiO2)", 3422: "FiO2 [measured]",
    # -- SYSTOLIC BLOOD PRESSURE Systolic/diastolic
    51: "Arterial BP [Systolic]", 442: "Manual BP [Systolic]", 455: "NBP [Systolic]",
    6701: "Arterial BP", 220179: "Non Invasive Blood Pr systolic", 220050: "Arterial Blood Pr systolic",
    3313: "BP Cuff [Systolic]",
    # -- DIASTOLIC BLOOD PRESSURE
    8368: "Arterial BP [Diastolic]", 8440: "Manual BP [Diastolic]", 8441: "NBP [Diastolic]",
    8555: "Arterial BP #2 [Diastolic]", 220180: "Non Invasive Blood Pr diastolic",
    220051: "Arterial Blood Pr diastolic", 225310: "ART BP Diastolic", 8502: "BP Cuff [Diastolic]",
    # -- MEAN ARTERIAL PRESSURE --Mean blood pressure
    456: "NBP Mean", 52: "Arterial BP Mean", 6702: "Arterial BP Mean #2", 443: "Manual BP Mean(calc)",
    220052: "Arterial Blood Pressure mean", 220181: "Non Invasive Blood Pressure mean", 225312: "ART BP mean",
    224: "IABP Mean", 224322: "IABP Mean", 3312: "BP Cuff [Mean]",
    # -- O2 saturation pulseoxymetry --SpO2 SPO2, peripheral
    646: "SpO2", 220277: "O2 saturation pulseoxymetry",
    # Capillary Refill[Rate]
    # 115:"Capillary Refill[Rate]",3348:"Capillary Refill[Rate]",8377:"Capillary Refill[Rate]",
    # -- Urine output
    40055: "Urine Out Foley outputevents", 43175: "Urine outputevents",
    40069: "Urine Out Void outputevents", 40094: "Urine Out Condom Cath outputevents",
    40715: "Urine Out Suprapubic outputevents", 40473: "Urine Out IleoConduit outputevents",
    40085: "Urine Out Incontinent outputevents", 40057: "Urine Out Rt Nephrostomy outputevents",
    40056: "Urine Out Lt Nephrostomy outputevents", 40405: "Urine Out Other outputevents",
    40428: "Orine Out Straight Cath outputevents", 40086: "Urine Out Incontinent outputevents",
    40096: "Urine Out Ureteral Stent #1 outputevents", 40651: "Urine Out Ureteral Stent #2 outputevents",
    226559: "Foley outputevents", 226560: "Void outputevents", 226561: "Condom Cath outputevents",
    226584: "Ileoconduit outputevents", 226563: "Suprapubic outputevents", 226564: "R Nephrostomy",
    226565: "L Nephrostomy outputevents", 226567: "Straight Cath outputevents",
    226557: "R Ureteral Stent outputevents", 226558: "L Ureteral Stent outputevents",
    227488: "GU Irrigant Volume In outputevents", 227489: "GU Irrigant/Urine Volume Out outputevents",
    # Glucose
    807: "Glucose", 811: "Glucose", 1529: "Glucose", 3745: "Glucose",
    3744: "Glucose", 225664: "Glucose", 220621: "Glucose", 226537: "Glucose",
    # WEIGHT
    762: "Weight Kg", 763: "Weight Kg", 3723: "Weight Kg", 3580: "Weight Kg", 226512: "Weight Kg",
    3581: "Weight lb", 3582: "Weight oz", 226531: "Admission Weight (lbs.)",
    # HEIGHT
    3485: "Length   Calc   (cm)", 4188: "Length in cm",
    920: "Admit Ht inches", 1394: "Height inches", 4187: "Length Calc inches", 3486: "Length in Inches",
    226707: "Height inches", 226730: "Height cm",
    # RR --Respiration Rate
    615: "Respiration Rate", 618: "Respiration Rate", 220210: "Respiration Rate",
    224690: "Respiration Rate", 614: "Resp Rate (Spont)", 224689: "Respiratory Rate (spontaneous)",
    651: "Spon RR (Mech.)", 224422: "Spont RR",
    # Respiratory Rate Set
    619: "Respiratory Rate Set", 224688: "Respiratory Rate (Set)",
    # O2 FLOW
    470: "O2 Flow (lpm)", 471: "O2 Flow (lpm)", 227287: "O2 Flow (additional cannula)", 223834: "O2 Flow",
    227582: "BiPap O2 Flow", 224691: "Flow Rate (L/min)",
}
drugs_list = {
    # Norepinephrine drug max  rate
    221906: "('Norepinephrine', 221906,): 89697 -statusdescription != 'Rewritten' ,mcg/kg/min",
    30120: "('Levophed-k', 30120, mcgkgmin): 476971",
    30047: "('Levophed', 30047 - wd.weight is null then rate / 80.0  or rate / wd.weight , mcgmin): 22272",
    # Epinephrine drug max  rate
    30119: "('Epinephrine-k', 30119, mcgkgmin): 82889",
    30044: "('Epinephrine', 30044, mcgmin , rate / wd.weight): 477",
    30309: "('Epinephrine drip', 30309,mcgkgmin) 104",
    221289: "('Epinephrine', 221289): 6413-statusdescription != 'Rewritten', mcg/kg/min",
    # Phenylephrine drug max  rate
    221749: "('Phenylephrine', 221749): 93571 -statusdescription != 'Rewritten',mcg/kg/min",
    30127: "('Neosynephrine', 30127,mcgmin, rate / wd.weight): 14317",
    30128: "('Neosynephrine-k:Phenylephrine', 30128,mcgkgmin): 554582",
    # Vasopressin drug max  rate
    222315: "('Vasopressin', 222315): 5648 - statusdescription != 'Rewritten',units/hour",
    30051: "('Vasopressin', 30051, units/hour): 165219",
    # Dopamine drug max  rate
    221662: "('Dopamine', 221662): 11389 - statusdescription != 'Rewritten',mcg/kg/min ",
    30043: "('Dopamine', 30043,mcgkgmin): 173745",
    30307: "('Dopamine drip', 30307,mcgkgmin): 32971",
    # Dobutamine drug max  rate
    221653: "('Dobutamine', 221653): 2233 - statusdescription != 'Rewritten',mcg/kg/min ",
    30042: "('Dobutamine', 30042,mcgkgmin): 66775",
    30306: "('Dobutamine drip', 30306,mcgkgmin): 1663",
    # Midazolam drug max  rate
    30124: "('Midazolam', 30124, mg/hour rate/weight---mcgkghr): 505509",
    221668: "('Midazolam (versed)', 221668): 71674 -statusdescription != 'Rewritten',mg/hour  rate/weight---mcgkghr",
    # Fentanyl drug max  rate
    221744: "('Fentanyl', 221744 mcg/hour): 86340 -statusdescription != 'Rewritten',mcg/hour,rate/weight--mcgkghr",
    225942: "('Fentanyl (concentrate)', 225942 mcg/hour): 45866 -statusdescription != 'Rewritten',mcg/hour,rate/weight--mcgkghr",
    30118: "('Fentanyl', 30118,mcg/hour rate/weight---mcgkghr): 780555",
    30308: "('Fentanyl drip', 30308, mcgkghr) 36595",
    30149: "('Fentanyl (conc)', 30149, ,mcg/hour rate/weight---mcgkghr) 35526",
    # Propofol  drug max  rate
    30131: "('Propofol', 30131,mcgkgmin): 924614",
    222168: "('Propofol', 222168): 178819 -statusdescription != 'Rewritten', 	mcg/kg/min",
    # Furosemide drug max  rate
    221794: "('Furosemide (lasix)', 221794, mg/hour - statusdescription != 'Rewritten'): 47444",
    30123: "(Furosemide 'Lasix', 30123, mg/hour): 193403",
    # Milrinone drug max  rate
    221986: "('Milrinone', 221986 - statusdescription != 'Rewritten', mcg/kg/min): 6249",
    30125: "('Milrinone', 30125,mcgkgmin): 132751",
    # Nitroglycerine  drug max  rate
    222056: "('Nitroglycerine', 222056 - statusdescription != 'Rewritten',mcg/kg/min): 31528",
    30121: "('Nitroglycerine-k', 30121,  mcgkgmin 	): 239268",
    30049: "('Nitroglycerine', 30049 / weight, mcgmin): 14398",
    # Nitroprusside  drug max  rate
    222051: "('Nitroprusside', 222051 - statusdescription != 'Rewritten',mcg/kg/min): 4462",
    30050: "('Nitroprusside', 30050,mcgkgmin): 112121",
    # Morphine sulfate  drug max  rate
    225154: "('Morphine sulfate', 225154 - statusdescription != 'Rewritten', mghr): 41064",
    30126: "('Morphine sulfate', 30126, mghr): 97439",
    # Lorazepam (ativan)  drug max  rate
    221385: "('Lorazepam (ativan)', 221385 - statusdescription != 'Rewritten',mghr): 18731",
    30141: "('Lorazepam Ativan', 30141,mghr): 173510",
    # Labetalol  drug max  rate
    225153: "('Labetalol', 225153- statusdescription != 'Rewritten',mg/min): 9206",
    30122: "('Labetolol', 30122, mg/min): 66164",
    # Diltiazem  drug max  rate
    221468: "('Diltiazem', 221468- statusdescription != 'Rewritten',mghr): 8765",
    30115: "('Diltiazem', 30115,mghr): 64405",
    # Argatroban drug  max rate
    225147: "('Argatroban', 225147, mcgkgmin , statusdescription != 'Rewritten') 1065",
    30173: "('Argatroban', 30173, mcgkgmin) 26166",
    # integrilin drug  max rate
    225151: "('Eptifibatide (integrilin)', 225151,mcgkgmin) 1264",
    30142: "('Integrelin', 30142,mcgkgmin): 34617",
    # Precedex drug  max rate
    225150: "('Dexmedetomidine (precedex)', 225150  statusdescription != 'Rewritten',mcg/kg/hour)",
    30167: "('Precedex', 30167, mcgkghr)",
    # Natrecor  drug  max rate
    222037: " ('Nesiritide', 222037,mg, statusdescription != 'Rewritten',mcg/kg/min) 35",
    30172: "('Natrecor', 30172,mcgkgmin): 51536",
    # Hydromorphone Dilaudid drug max rate
    221833: "('Hydromorphone (Dilaudid)', 221833- statusdescription != 'Rewritten',mg/hour 	 ): 32461",
    30163: "('Hydromorphone (Dilaudid)', 30163,mg/hour): 41012",
    # Nicardipine drug max rate
    222042: "('Nicardipine', 222042- statusdescription != 'Rewritten',mcgkgmin): 13611",
    30178: "('Nicardipine', 30178,mcgkgmin): 32318",
    # Esmolol drug max rate
    221429: "('Esmolol', 221429- statusdescription != 'Rewritten',mcg/kg/min): 4888",
    30117: "('Esmolol', 30117,mcgkgmin): 28666",
}
