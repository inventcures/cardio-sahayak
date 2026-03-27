DOCTOR_DISCLAIMER = (
    "Cardio-Sahayak is a clinical decision support tool. All recommendations "
    "require physician confirmation. This system does not provide autonomous "
    "diagnosis. Guidelines: IJAM 2023 Consensus, ESC 2019, Indian National "
    "Consensus on Cardiology."
)

PATIENT_DISCLAIMER_EN = (
    "This tool provides general health information only. It is NOT a substitute "
    "for professional medical advice, diagnosis, or treatment. Always consult "
    "your doctor before making any health decisions."
)

PATIENT_DISCLAIMER_HI = (
    "Yah upkaran kewal samanya swasthya jaankari pradaan karta hai. Yah "
    "vyavsayik chikitsa salah, nidaan, ya upchar ka vikalp NAHI hai. Koi bhi "
    "swasthya nirnay lene se pehle hamesha apne doctor se salah lein."
)

CHW_DISCLAIMER = (
    "This is a screening tool for community health workers. All RED and YELLOW "
    "risk patients MUST be referred to a medical facility. This tool does not "
    "replace clinical judgment."
)

REGULATORY_NOTICE = (
    "Cardio-Sahayak is intended for use as a Class B Software as a Medical "
    "Device (SaMD) under CDSCO regulations, requiring physician oversight. "
    "Not approved for autonomous clinical decision-making."
)

CDSCO_CLASSIFICATION = {
    "class": "B",
    "risk_level": "medium",
    "requires_physician_oversight": True,
    "regulatory_body": "CDSCO (Central Drugs Standard Control Organisation)",
    "applicable_rules": "Medical Devices Rules 2017",
    "quality_standard": "ISO 13485",
    "risk_management": "ISO 14971",
}

DATA_PRIVACY_NOTICE = (
    "Patient data is processed locally and not transmitted to external servers "
    "in edge deployment mode. Data handling complies with the Digital Personal "
    "Data Protection Act (DPDPA) 2023. Patients have the right to request "
    "deletion of their data."
)
