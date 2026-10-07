import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump


# ============================================================
# 1. CREATE COLLEGE MESSAGE DATASET
# ============================================================

academic_messages = [
    "Mid semester examinations will begin from Monday.",
    "Students are requested to submit their assignments before Friday.",
    "The mathematics department is conducting a workshop on Python.",
    "The internal assessment timetable has been released.",
    "The database management systems lab will be conducted tomorrow.",
    "Students must complete the online academic registration.",
    "The computer networks lecture has been rescheduled to 11 AM.",
    "The university examination form submission deadline is Friday.",
    "Attendance shortage students should meet the department coordinator.",
    "The physics practical examination will be held next week.",
    "The project review is scheduled for Wednesday.",
    "Students should submit their project reports to the department.",
    "A seminar on artificial intelligence will be conducted tomorrow.",
    "The syllabus for the upcoming examination has been uploaded.",
    "The assignment submission portal will close tonight.",
    "The faculty has announced a revision session for machine learning.",
    "Students are requested to check the updated examination timetable.",
    "The college is conducting a workshop on data science.",
    "The viva examination schedule has been announced.",
    "The last date for examination form submission is approaching.",
    "The operating systems test will be conducted on Friday.",
    "Students should bring their practical journals for verification.",
    "The department has announced a coding workshop.",
    "The academic calendar for the semester is now available.",
    "The seminar attendance is compulsory for all students.",
    "The project presentation will be conducted in the seminar hall.",
    "Students must complete their laboratory records.",
    "The engineering mathematics tutorial will be held after lunch.",
    "The department has scheduled an extra lecture for tomorrow.",
    "The internal marks will be displayed on the notice board.",
    "Students can download the new academic timetable.",
    "The faculty has announced a doubt clearing session.",
    "The end semester examination timetable has been published.",
    "The college will conduct a technical workshop this weekend.",
    "Students should verify their examination seat numbers.",
    "The practical examination instructions have been shared.",
    "The department is organizing a research methodology seminar.",
    "The assignment evaluation will begin next week.",
    "Students are requested to attend the guest lecture.",
    "The final year project submission date has been extended.",
    "The college has announced a scholarship application deadline.",
    "Eligible students can apply for the merit scholarship.",
    "Scholarship documents must be submitted to the office.",
    "The scholarship verification process starts tomorrow.",
    "Students should submit their income certificate for scholarship verification.",
    "The university has released scholarship guidelines.",
    "The scholarship application portal is now open.",
    "Students can contact the academic office for scholarship queries.",
    "The scholarship renewal deadline is next week.",
    "The department has announced a faculty development workshop.",
    "A technical lecture on cloud computing is scheduled tomorrow.",
    "Students should register for the upcoming academic workshop.",
    "The college library will conduct a research orientation session.",
    "The statistics examination will be held on Tuesday.",
    "The project synopsis submission deadline is Friday.",
    "The department has published the laboratory examination schedule.",
    "Students are requested to check the revised lecture timetable.",
    "The college has scheduled a remedial class for mathematics.",
    "The assignment guidelines have been uploaded to the portal.",
    "The examination hall instructions have been circulated.",
    "Students must carry their identity cards during examinations.",
    "The department has announced an academic counseling session.",
    "The attendance report will be updated this week.",
    "The semester registration process starts Monday.",
    "Students should complete course registration before the deadline.",
    "The faculty has uploaded study material for the examination.",
    "The college is organizing a workshop on cybersecurity.",
    "The department has announced a seminar on machine learning.",
    "Students can register for the upcoming technical lecture.",
    "The practical timetable has been updated.",
    "The university has announced the examination results.",
    "Students should collect their examination hall tickets.",
    "The college has released the semester academic schedule.",
    "The project evaluation will be conducted next week.",
    "Students must submit their laboratory assignments.",
    "The department has announced an extra tutorial class.",
    "The college is conducting an academic orientation program.",
    "Students are requested to attend the compulsory revision lecture.",
]


placement_messages = [
    "Infosys is conducting a campus placement drive next week.",
    "Students interested in the placement drive should register today.",
    "The placement cell has announced an aptitude test.",
    "TCS will conduct an online assessment for eligible students.",
    "The recruitment drive for software developers is scheduled tomorrow.",
    "Eligible final year students can apply for the internship opportunity.",
    "The placement cell has released the interview schedule.",
    "Students shortlisted for the company interview should report at 9 AM.",
    "A pre-placement talk will be conducted by the company today.",
    "The company has announced openings for software engineering interns.",
    "Students must upload their resumes before the placement deadline.",
    "The aptitude test for campus recruitment will be held Friday.",
    "The placement department has published the shortlisted candidates.",
    "An internship opportunity is available for machine learning students.",
    "The recruitment team will visit the college next week.",
    "Students should register on the placement portal.",
    "The technical interview round is scheduled for Monday.",
    "The HR interview schedule has been shared by the placement cell.",
    "Eligible students can apply for the summer internship program.",
    "The company will conduct a coding assessment tomorrow.",
    "The placement office has announced a new recruitment drive.",
    "Students shortlisted for the aptitude round will receive an email.",
    "The internship application deadline is Friday.",
    "The placement cell is organizing a resume building workshop.",
    "A multinational company is visiting campus for recruitment.",
    "Students must carry their resumes for the interview.",
    "The campus recruitment registration portal is now open.",
    "The company has announced software developer positions.",
    "Students from the computer department can apply for this drive.",
    "The placement cell has scheduled a mock interview session.",
    "The coding test results have been announced.",
    "Shortlisted students should prepare for the technical interview.",
    "A pre-placement session will be held in the seminar hall.",
    "The company has released eligibility criteria for recruitment.",
    "Students with the required CGPA can register for the drive.",
    "The placement coordinator has shared interview instructions.",
    "The internship selection process will begin next week.",
    "A technology company is hiring engineering graduates.",
    "The recruitment test will include aptitude and coding questions.",
    "Students should update their resumes before applying.",
    "The placement cell has announced an internship fair.",
    "The company presentation will begin at 2 PM.",
    "Selected students will receive joining instructions from the company.",
    "The final interview round is scheduled tomorrow.",
    "The placement office has published the recruitment calendar.",
    "Students can apply for the software engineer position.",
    "The company requires candidates with basic programming skills.",
    "The aptitude preparation session will be conducted this evening.",
    "The placement cell is organizing a career guidance seminar.",
    "Students should check their placement eligibility status.",
    "The company has announced vacancies for data analyst interns.",
    "The recruitment registration deadline has been extended.",
    "Shortlisted candidates must attend the pre-placement talk.",
    "The technical assessment link has been shared with candidates.",
    "The company will conduct interviews through the college placement cell.",
    "Students interested in internships should contact the placement office.",
    "The placement coordinator has announced a new internship opportunity.",
    "Candidates must submit their resumes in PDF format.",
    "The campus drive will include aptitude, technical, and HR rounds.",
    "Students selected for the next round will be notified soon.",
    "The company has announced a recruitment drive for freshers.",
    "The placement office has shared company eligibility requirements.",
    "A coding challenge will be conducted as part of the recruitment process.",
    "The interview venue has been changed to the main seminar hall.",
    "The company will conduct a virtual recruitment drive.",
    "Students should register before the placement portal closes.",
    "The placement cell has announced an internship for AI students.",
    "Eligible students can attend the company pre-placement presentation.",
    "The final shortlist for the recruitment drive has been released.",
    "The company has invited applications for graduate trainee roles.",
    "Students must attend the placement orientation session.",
    "The technical test will be conducted online.",
    "The placement cell has shared the interview preparation material.",
    "Students selected for the internship will receive offer letters.",
    "The recruitment drive is open to eligible engineering students.",
    "The company has scheduled an online aptitude assessment.",
    "The placement office has announced a new campus opportunity.",
]


promotional_messages = [
    "Get 50 percent off on all college stationery today!",
    "Buy one notebook and get another one free.",
    "Special discount available on backpacks this weekend.",
    "Students get exciting offers on headphones today.",
    "Flash sale starts at midnight. Shop now!",
    "Get amazing discounts on laptops this week.",
    "Exclusive student offer available at our store.",
    "Buy two T-shirts and get one free.",
    "Flat 40 percent discount on selected products.",
    "Limited time offer on college accessories.",
    "Use our student coupon and save more.",
    "Weekend sale is now live. Don't miss out!",
    "Special food festival discount for students.",
    "Get a free drink with every combo purchase.",
    "Mega sale on shoes starts today.",
    "Exclusive discount available for college students.",
    "Order today and receive free delivery.",
    "Limited time shopping offer ends tonight.",
    "Save big on electronics during our festival sale.",
    "Student discount available on all selected items.",
    "Grab your favorite products at special prices.",
    "Flash discount available for the next 24 hours.",
    "Get 30 percent off on all accessories.",
    "Special offer: buy three items and get one free.",
    "Festival sale is now open!",
    "Shop today and get an additional discount.",
    "Free shipping is available for a limited time.",
    "Special weekend deal for students.",
    "Get exclusive offers through our mobile app.",
    "Mega discount on fashion products today.",
    "Limited stock available at discounted prices.",
    "Use coupon code SAVE20 for an extra discount.",
    "Special student deal available until Sunday.",
    "Enjoy exciting offers on food and beverages.",
    "Get your favorite snacks at special prices.",
    "Early bird discount available for today's event.",
    "Buy now and save more with our student offer.",
    "Summer sale has started with huge discounts.",
    "Get special prices on selected electronic products.",
    "Flash sale: discounts available for a few hours.",
    "Special offer for new customers.",
    "Save up to 60 percent during our mega sale.",
    "Exclusive online shopping deal is live.",
    "Get a special discount on your first order.",
    "Free gift with selected purchases today.",
    "Don't miss our weekend shopping festival.",
    "Limited time offer on college merchandise.",
    "Get special prices on bags and accessories.",
    "Student coupon available for online orders.",
    "Big savings are waiting for you this weekend.",
    "Special combo offer available today.",
    "Buy more and save more with our latest deal.",
    "Festival discounts are available now.",
    "Get exciting deals on sports accessories.",
    "Special offer available only for today.",
    "Enjoy a discount on every food combo.",
    "Free delivery offer ends tonight.",
    "Shop our new collection with student discounts.",
    "Exclusive deal for members of our shopping club.",
    "Get a free accessory with selected purchases.",
    "Huge savings on selected products this week.",
    "Limited period discount is now available.",
    "Special deal available through our website.",
    "Save money with our latest promotional offer.",
    "Get exciting prices during our clearance sale.",
    "Weekend flash sale is starting soon.",
    "Special student pricing is available now.",
    "Order now and enjoy a special discount.",
    "Get extra savings using our promotional coupon.",
    "New customer offer available for a limited time.",
    "Shop today to enjoy festival discounts.",
    "Exclusive food combo offer for students.",
    "Special prices available while stocks last.",
    "Grab this limited time deal before it ends.",
    "Amazing discounts are available on selected items.",
    "Get special shopping vouchers today.",
    "Our latest promotional campaign is now live.",
]


# ============================================================
# 2. CREATE DATAFRAME
# ============================================================

data = []

for message in academic_messages:
    data.append([message, "Academic"])

for message in placement_messages:
    data.append([message, "Placement"])

for message in promotional_messages:
    data.append([message, "Promotional"])


df = pd.DataFrame(data, columns=["message", "label"])

# Remove accidental duplicates
df = df.drop_duplicates().reset_index(drop=True)

print("Total messages:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())


# Save dataset
df.to_csv("college_messages.csv", index=False)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. BUILD NAIVE BAYES PIPELINE
# ============================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "naive_bayes",
        MultinomialNB()
    )
])


# ============================================================
# 5. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 6. EVALUATE MODEL
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 7. SAVE MODEL
# ============================================================

dump(model, "college_message_model.joblib")

print("\nModel saved as:")
print("college_message_model.joblib")


# ============================================================
# 8. TEST SAMPLE MESSAGES
# ============================================================

sample_messages = [
    "The semester examination timetable has been released.",
    "Students can register for the upcoming campus recruitment drive.",
    "Get 50 percent off on all headphones today!"
]

print("\nSample Predictions:")

for message in sample_messages:
    prediction = model.predict([message])[0]
    print(f"\nMessage: {message}")
    print(f"Prediction: {prediction}")