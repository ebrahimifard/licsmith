# 🧾 licsmith

**`licsmith`** is a Python toolkit that helps developers collect and analyse software licenses in Python projects.  
It simplifies open-source compliance by automatically discovering, aggregating, and interpreting license files for all dependencies.

> 🚧 **Status:** Work in progress (still under active development).  
> A PyPI-installable package is coming soon!

---

## ✨ Features

### 🧩 1. License Collector

Automatically finds all external dependencies of your Python project, retrieves their license files, and aggregates them into a single license report for easy review and compliance tracking.

#### ▶️ Run

```bash
python license_collector.py
````

#### 🧭 Example CLI Session

```
Welcome to License Collector!
==================================================
This tool identifies external packages, collects their corresponding license files,
and aggregates them into a single report.

Please enter the path to the root of your project folder.
For example:

    my_project/
    ├── src/
    │   └── main.py
    ├── data/
    │   └── input.csv
    └── README.md

In Windows => C:\Users\YourName\Documents\my_project\
In Linux/Mac => /home/yourname/projects/my_project/

Path: /home/user/projects/my_project
Please enter the name of the aggregated license file (e.g., LICENSES.txt): LICENSES.txt
```

🪄 **What it does**

* Detects installed dependencies.
* Extracts their license text or metadata.
* Aggregates all licenses into one file (e.g., `LICENSES.txt`).

---

### ⚖️ 2. License Analyser (LLM-Assisted)

Reads and interprets a given license file using a Large Language Model (LLM).
It answers key legal and compliance questions with a short verdict and an explanation citing relevant parts of the license.

#### ▶️ Run

```bash
python license_analyser.py
```

#### 🧭 Example CLI Session

```
Enter path to LICENSE file: /home/user/my_project/LICENSE
Do you want to save the results? [Y/N]
Y
Enter path you like to save the results: /home/user/my_project/results/
✅ Analysis complete!
```

---

### 💬 Example Output

```
Question: What is the name/type of the license (MIT, Apache 2.0, GPLv3, BSD, proprietary, etc.)?

Answer: Apache 2.0

Explanation: The license text begins with "Apache License, Version 2.0, January 2004,"
clearly identifying the name and version of the license.
==================================================
Question: Is it a standard open-source license or a custom license?

Answer: Yes.

Explanation: The provided text is the standard Apache License 2.0,
a well-known and widely recognized open-source license.
It includes all the standard provisions of the Apache License,
such as grants of copyright and patent rights,
redistribution conditions, disclaimer of warranty,
and limitation of liability. The structure, terminology,
and content match the official Apache License 2.0
as published by the Apache Software Foundation.
==================================================
```

Each question is answered in this consistent, human-readable format with a
short verdict first, followed by a supporting rationale.

---

### 🧾 Questions Answered

The analyser currently answers around 25 core compliance questions, grouped by topic:

#### 🪪 Identification

* What is the name/type of the license (MIT, Apache 2.0, GPLv3, etc.)?
* Is it a standard open-source license or custom?
* What is the copyright notice?

#### ✅ Permissions

* Does it allow commercial use?
* Can it be used privately or internally?
* Can the code be modified and redistributed?
* Is sublicensing permitted?

#### ⚙️ Conditions

* Is attribution required?
* Must I include a copy of the license?
* Must I disclose modifications?
* Are there notice requirements?

#### 🔄 Copyleft & Disclosure

* Are there copyleft obligations?
* Must derivative works use the same license?
* Do I need to disclose source code changes?

#### ⚠️ Limitations & Liability

* Does the license disclaim warranty?
* Does it limit or deny author liability?

#### 🧠 Patents & Restrictions

* Are there patent grants or restrictions?
* Are there field-of-use or jurisdiction clauses?
* Are there extra permissions or restrictions?

#### 💼 Practical Use

* Can I use this in proprietary/commercial software?
* Do I need to make my own code open-source?
* What attribution text must I include?
* Do I need to track license compatibility?

---

## 🌟 Stay Tuned

`licsmith` is evolving into a comprehensive **license compliance assistant** for Python developers.
⭐ Star the repository to get updates as new features are released!
