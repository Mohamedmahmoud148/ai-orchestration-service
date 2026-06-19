"""
app/agents/intent_examples.py  —  Intent Example Bank

Ground-truth examples used to build the per-intent embedding centroids
for the Layer-1 EmbeddingIntentClassifier.

Design rules:
  - 10–20 examples per intent (more = smoother centroid)
  - Balanced across Arabic / English / mixed / Arabizi (after Layer-0 transliteration)
  - Examples should be NATURAL, not triggers — real things a user would type
  - Do NOT add overly-specific examples; centroids work on semantic similarity,
    not substring matching
  - To add examples: append to the list and restart — warm_up() re-embeds on boot

Maintenance:
  When you add a new intent to VALID_INTENTS in planner.py, add a key here too.
  The classifier will log a WARNING if it finds an intent with < 5 examples.
"""
from __future__ import annotations

INTENT_EXAMPLES: dict[str, list[str]] = {

    # ── general_chat ──────────────────────────────────────────────────────────
    "general_chat": [
        "hello",
        "hi there",
        "how are you",
        "what can you do",
        "tell me something interesting",
        "thanks",
        "thank you so much",
        "مرحبا",
        "ازيك",
        "كيف حالك",
        "ايه اللي تقدر تعمله",
        "شكراً",
        "تمام يسلموا",
        "صباح الخير",
        "good morning",
        "what's up",
        "explain what machine learning is",
        "ما هو الذكاء الاصطناعي",
        "اشرحلي مفهوم البرمجة",
    ],

    # ── backend_api_query ─────────────────────────────────────────────────────
    "backend_api_query": [
        # English — counts and lists
        "how many students are in computer science",
        "show me all doctors in the math department",
        "list all available courses",
        "who teaches algorithms",
        "what departments exist",
        "how many colleges are there",
        "total number of students enrolled",
        # English — my data
        "who am i",
        "what is my name",
        "show my profile",
        "my college",
        "my department",
        "what batch am i in",
        # Arabic — counts and lists
        "كم عدد الطلاب في قسم الحاسبات",
        "اعرضلي كل الدكاترة في القسم",
        "ايه الكليات الموجودة",
        "كام دكتور عندنا",
        "قائمة المواد المتاحة",
        "الأقسام اللي في الجامعة",
        # Arabic — my data
        "انا مين",
        "اسمي ايه",
        "معلوماتي",
        "بياناتي",
        "كليتي ايه",
        "انا في انهي قسم",
        "عرفني بنفسك",
        # Doctor specific
        "درجات الطلاب في المادة",
        "نتايج الطلاب",
        "كام طالب مسجل عندي",
        "الطلاب اللي رسبوا",
        "student grades for my course",
        "how many students are registered in my subject",
        # Schedule queries — MUST route to backend_api_query (not study_plan or general_chat)
        "what's my schedule this week",
        "show me my class schedule",
        "what classes do I have today",
        "do I have classes tomorrow",
        "when is my next lecture",
        "what is my timetable",
        "my weekly schedule",
        "جدولي الدراسي",
        "جدولي الأسبوعي",
        "ايه جدولي النهارده",
        "عندي محاضرات بكرا",
        "جدول المحاضرات",
        "متى موعد محاضرتي",
        "ايه المواد اللي عندي الأسبوع ده",
        "ايه المحاضرات دلوقتي",
        "الجدول الدراسي",
        "جدول الأسبوع",
        "my schedule",
        "show schedule",
        "my lectures",
        "class timetable",
        "weekly timetable",
    ],

    # ── generate_exam ─────────────────────────────────────────────────────────
    "generate_exam": [
        # English
        "generate an exam for database subject",
        "create a midterm test for operating systems",
        "make 20 questions for algorithms course",
        "build a final exam for computer networks",
        "write a quiz about data structures",
        "prepare a test for my students",
        "design an exam with MCQ questions",
        "produce an exam on machine learning",
        "set a midterm for software engineering",
        # Arabic
        "اعمل امتحان في قواعد البيانات",
        "انشئ اختبار للطلاب في مادة الشبكات",
        "عايز امتحان لمادة الخوارزميات",
        "حضرلي امتحان نص الترم",
        "اكتبلي اسئلة امتحان للمادة",
        "سوي امتحان في الذكاء الاصطناعي",
        "صمم اختبار للطلاب",
        "جهز امتحان نهايه الترم",
        "عمل امتحان لمادة الكمبيوتر",
        # Mixed
        "اعملي quiz عن machine learning",
        "generate exam في database",
        "create امتحان for operating systems",
        "اعملي midterm في الـ networks",
        # Arabizi (pre-transliterated by Layer 0)
        "عايز exam في الـ AI",
        "اعملي exam عن networks",
    ],

    # ── result_query ──────────────────────────────────────────────────────────
    "result_query": [
        # English
        "show me my exam results",
        "what is my current GPA",
        "show my transcript",
        "did I pass algorithms",
        "my grades for this semester",
        "what grades did I get",
        "show my academic record",
        # Arabic
        "اعرضلي نتيجتي",
        "ايه معدلي الحالي",
        "كشف الدرجات بتاعي",
        "هل نجحت في الخوارزميات",
        "درجاتي في الترم ده",
        "ايه درجاتي",
        "نتيجتي في الامتحان",
        "سجل درجاتي",
        "نسبتي الكلية كام",
    ],

    # ── complaint_submit ──────────────────────────────────────────────────────
    "complaint_submit": [
        # English
        "I want to complain about the doctor",
        "the exam was unfair",
        "my grade is incorrect",
        "I have a complaint to submit",
        "the doctor gave me the wrong grade",
        "I want to report an issue with the exam",
        "file a complaint against the professor",
        # Arabic
        "عايز اشتكي من الدكتور",
        "الامتحان كان ظالم",
        "درجتي غلطه",
        "عندي شكوى",
        "الدكتور خصملي درجات بدون سبب",
        "ابلغ عن مشكله في الامتحان",
        "ارفع شكوى",
        # Mixed
        "submit complaint للدكتور",
        "اشتكي من الـ doctor",
        # Arabizi (pre-transliterated by Layer 0)
        "عندي مشكله مع الدكتور",
        "ارفع شكوى للادارة",
    ],

    # ── complaint_summary ─────────────────────────────────────────────────────
    "complaint_summary": [
        "show me all student complaints",
        "summarize the complaints this month",
        "how many complaints were filed",
        "list pending complaints",
        "review the complaint reports",
        "عرضلي الشكاوى المقدمة",
        "ملخص الشكاوى",
        "كام شكوى اتقدمت",
        "الشكاوى اللي لسه مش متعاملين معاها",
        "تقرير الشكاوى",
    ],

    # ── academic_advice ───────────────────────────────────────────────────────
    "academic_advice": [
        # English
        "what courses should I take next semester",
        "how can I improve my academic standing",
        "give me advice about my studies",
        "what should I focus on this year",
        "am I on track to graduate",
        "what are my weakest subjects",
        "how is my academic performance",
        # Arabic
        "ايه المواد المناسبة ليا الترم الجاي",
        "ازاي أحسن وضعي الأكاديمي",
        "نصيحة أكاديمية ليا",
        "ايه اللي لازم أركز عليه",
        "هل أنا في المسار الصح",
        "ايه نقاط ضعفي",
        "ايه وضعي الاكاديمي",
        "هل هكمل كويس",
    ],

    # ── study_plan ────────────────────────────────────────────────────────────
    "study_plan": [
        # English
        "make me a study plan for midterms",
        "how should I study this week",
        "create a revision schedule",
        "what should I study today",
        "help me prepare for the final exam",
        "give me a weekly study timetable",
        "how to prepare for my exams",
        "prioritize my subjects for exam week",
        "exam preparation strategy",
        # Arabic
        "اعمللي خطة مذاكرة",
        "كيف أذاكر للميدتيرم",
        "جدول مذاكرة لأسبوع",
        "اذاكر ايه النهارده",
        "خطة مذاكرة للامتحانات",
        "ساعدني أذاكر",
        "رتبلي أولويات المذاكرة",
        "ازاي أستعد للامتحان",
        "جدول مراجعة للـ final",
    ],

    # ── regulation ────────────────────────────────────────────────────────────
    "regulation": [
        # English
        "explain the academic regulation",
        "what subjects are required in year one",
        "what are the graduation requirements",
        "how many credit hours do I need to graduate",
        "what is in the academic study plan",
        "subjects in the first semester",
        "curriculum requirements",
        # Arabic
        "اشرح اللائحة الأكاديمية",
        "مواد السنة الأولى",
        "متطلبات التخرج",
        "ايه اللي في اللائحة",
        "مواد الترم الأول",
        "كم ساعة محتاج للتخرج",
        "الخطة الدراسية",
        "مواد سنة تانية في اللائحة",
        "شروط التخرج",
        "دليل الطالب",
    ],

    # ── material_explanation ──────────────────────────────────────────────────
    "material_explanation": [
        # English
        "explain the data structures course",
        "summarize the machine learning material",
        "what is this subject about",
        "give me an overview of the algorithms course",
        "help me understand operating systems",
        "break down the computer networks material",
        "what topics are covered in this course",
        # Arabic
        "اشرح مادة قواعد البيانات",
        "لخصلي مادة الذكاء الاصطناعي",
        "ايه محتوى المادة دي",
        "شرح موضوع الـ sorting algorithms",
        "فهمني مادة الشبكات",
        "ايه اللي بتتكلم عنه المادة دي",
        "اشرحلي محتوى كورس البرمجة",
        "عايز ملخص للمادة",
    ],

    # ── material_qa ───────────────────────────────────────────────────────────
    "material_qa": [
        "what does the lecture say about binary trees",
        "according to the course material what is normalization",
        "from the lecture what is the difference between TCP and UDP",
        "based on the material explain recursion",
        "what does the textbook say about sorting",
        "من المحاضرة ايه الفرق بين الـ stack والـ queue",
        "من المادة اشرح الـ normalization",
        "في الكتاب ايه اللي بيقوله عن الـ binary search",
        "من الملزمة اشرحلي الـ linked list",
        "من المحاضرة ايه هو الـ deadlock",
    ],

    # ── action_execute ────────────────────────────────────────────────────────
    "action_execute": [
        # English enrollment
        "enroll me in all available courses",
        "register me for this semester",
        "sign me up for the AI course",
        "auto enroll me in my subjects",
        "register me for all courses",
        # Arabic enrollment
        "سجلني في المواد",
        "اسجلني في الترم",
        "عايز أسجل في كل المواد",
        "سجل لي في مواد السنة دي",
        "تسجيل المواد بتاعتي",
        "ابدأ التسجيل",
        "سجلني في كل حاجة",
        # Admin actions
        "add a new student",
        "create a doctor account",
        "أضف طالب جديد",
        "أنشئ حساب دكتور",
    ],

    # ── assignment_query ──────────────────────────────────────────────────────
    "assignment_query": [
        # English
        "show me my assignments",
        "when is the assignment due",
        "did I submit the homework",
        "what are my pending assignments",
        "assignment deadline for databases",
        "show assignment status",
        "have I submitted all assignments",
        # Arabic
        "اعرضلي واجباتي",
        "امتى موعد التسليم",
        "سلمت الواجب ولا لأ",
        "الواجبات اللي عليا",
        "امتى آخر يوم تسليم",
        "الواجبات المتأخرة",
        "ايه الواجب الجاي",
    ],

    # ── summarization ─────────────────────────────────────────────────────────
    "summarization": [
        "summarize this document",
        "give me a summary of this text",
        "make a brief summary",
        "condense this into key points",
        "لخص هذا المستند",
        "عمل ملخص لهذا النص",
        "اختصر الموضوع ده",
        "ايه أهم نقاط في الموضوع ده",
    ],

    # ── file_extraction ───────────────────────────────────────────────────────
    "file_extraction": [
        "extract information from this file",
        "read this PDF and tell me what's in it",
        "get the data from this uploaded file",
        "parse this document",
        "استخرج المعلومات من الملف",
        "اقرا الـ PDF ده وقولي فيه ايه",
        "احضر البيانات من الملف ده",
    ],

    # ── file_processing ───────────────────────────────────────────────────────
    "file_processing": [
        "upload student grades from excel",
        "bulk upload the students list",
        "process the grades file",
        "upload the Excel file with students",
        "ارفع درجات الطلاب من الـ Excel",
        "رفع قائمة الطلاب بالجملة",
        "معالجة ملف الدرجات",
    ],

    # ── cv_analysis ───────────────────────────────────────────────────────────
    "cv_analysis": [
        "analyze my CV",
        "review my resume",
        "give me feedback on my CV",
        "what skills do I have according to my CV",
        "help me improve my resume",
        "حلل الـ CV بتاعي",
        "راجع السيرة الذاتية بتاعتي",
        "ايه نقاط قوتي في الـ CV",
        "ساعدني أحسن الـ resume بتاعي",
    ],

    # ── AI Companion Platform intents ────────────────────────────────────────

    "academic_coach": [
        "how is my academic performance",
        "analyze my grades",
        "what are my weak subjects",
        "am I at academic risk",
        "give me a full academic analysis",
        "coach me on my studies",
        "كيف وضعي الأكاديمي",
        "حلل درجاتي",
        "ايه نقاط ضعفي الأكاديمية",
        "هل أنا في خطر أكاديمي",
        "تحليل أكاديمي شامل",
        "كوّحني في المذاكرة",
    ],

    "quiz_me": [
        "quiz me on this topic",
        "test me on databases",
        "ask me questions about algorithms",
        "I want to practice with questions",
        "interactive quiz session",
        "سألني على هذا الموضوع",
        "اختبرني في قواعد البيانات",
        "اسألني أسئلة على الخوارزميات",
        "عايز أتمرن بأسئلة",
        "جلسة quiz",
    ],

    "generate_flashcards": [
        "generate flashcards for binary trees",
        "make flashcards about sorting algorithms",
        "create revision cards for this topic",
        "flashcard deck for databases",
        "اعمل flashcards على الـ binary trees",
        "بطاقات مراجعة لموضوع التشفير",
        "اعملي flashcards على قواعد البيانات",
        "بطاقات حفظ للموضوع ده",
    ],

    "generate_examples": [
        "give me practical examples for recursion",
        "examples of polymorphism in Java",
        "show me real-world examples of normalization",
        "أعطني أمثلة عملية على الـ recursion",
        "أمثلة على الـ polymorphism",
        "أمثلة واقعية على الـ normalization",
    ],

    "generate_exercises": [
        "generate practice exercises for sorting",
        "give me exercises on SQL queries",
        "create practice problems for calculus",
        "اعمل تمارين تدريبية على الـ sorting",
        "مسائل تدريبية على SQL",
        "تمارين على التفاضل والتكامل",
    ],

    "progress_report": [
        "show me my weekly progress report",
        "how did I do this week",
        "monthly academic report",
        "analyze my study habits",
        "how consistent have I been",
        "تقرير أسبوعي",
        "ايه اللي عملته الأسبوع ده",
        "تقرير شهري للمذاكرة",
        "حلل عادات مذاكرتي",
        "هل بذاكر بانتظام",
    ],

    "learning_assistant": [
        "explain recursion with examples",
        "help me understand linked lists",
        "generate a summary of chapter 3",
        "create memory tricks for sorting algorithms",
        "help me memorize these concepts",
        "اشرحلي الـ recursion بأمثلة",
        "ساعدني أفهم الـ linked lists",
        "عمل ملخص للفصل الثالث",
        "حيل حفظ لخوارزميات الترتيب",
        "ساعدني أحفظ المفاهيم دي",
    ],

    "doctor_analytics": [
        "show me class performance analytics",
        "class performance breakdown for my subjects",
        "overview of all my courses",
        "how is my class doing overall",
        "teaching dashboard",
        "عرضلي تحليل أداء الفصل",
        "ملخص أداء كل المواد",
        "كيف فصلي بيعمل",
        "لوحة تحكم التدريس",
        "عرض أداء طلابي",
    ],

    "doctor_risk_students": [
        "show me students at risk",
        "which students are failing",
        "at-risk student list",
        "students who might fail",
        "who needs intervention",
        "الطلاب اللي في خطر",
        "مين محتاج مساعدة",
        "الطلاب اللي ممكن يرسبوا",
        "قائمة الطلاب في خطر",
        "مين محتاج تدخل",
    ],

    "doctor_weak_topics": [
        "which topics are students struggling with",
        "weak topics in my course",
        "most difficult exam topics",
        "where are students making mistakes",
        "topic performance analysis",
        "أي موضوع الطلاب بيعانوا فيه",
        "المواضيع الصعبة في المادة",
        "أكثر موضوع فيه أخطاء",
        "تحليل المواضيع الضعيفة",
        "وين الطلاب بيغلطوا",
    ],

    "doctor_recommendations": [
        "what should I focus on as a teacher",
        "AI teaching recommendations",
        "what actions should I take",
        "teaching suggestions",
        "how to improve my class",
        "ايه اللي المفروض أعمله دلوقتي",
        "توصيات تدريسية",
        "إيه التوصيات بتاعت الـ AI",
        "ازاي أحسن أداء فصلي",
        "خطوات تحسين التدريس",
    ],
}


# ── Validation helper ─────────────────────────────────────────────────────────

def validate_examples() -> list[str]:
    """
    Check the example bank and return a list of warning messages.
    Called at startup to catch misconfiguration early.
    """
    warnings: list[str] = []
    for intent, examples in INTENT_EXAMPLES.items():
        if len(examples) < 5:
            warnings.append(
                f"Intent '{intent}' has only {len(examples)} examples — "
                f"add at least 5 for reliable centroid quality."
            )
    return warnings
