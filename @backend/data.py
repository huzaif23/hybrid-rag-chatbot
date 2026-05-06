"""
IRS Tax Dataset - indexed chunks for RAG retrieval (not for mock answers).
"""

IRS_DATASET = [
    {
        "content": """The Form 1040 is used to file federal income taxes. Follow these steps to file:
1. Gather W-2 and 1099 forms
2. Determine your filing status (Single, Married Filing Jointly, Married Filing Separately, Head of Household)
3. Calculate your taxable income
4. Complete Form 1040
5. Pay any tax due or claim a refund
6. File electronically (e-file) or mail Form 1040
7. Expect processing time (7-21 days for e-file)
8. File by April 15th for most taxpayers""",
        "source": "https://www.irs.gov/instructions/i1040",
        "form": "1040",
        "topic": "filing_steps",
    },
    {
        "content": """Standard vs Itemized Deductions:
- Standard Deduction 2024: $14,600 (Single), $27,700 (Married Filing Jointly)
- Itemized Deductions include:
  * Mortgage Interest (Schedule A)
  * State and Local Taxes (capped at $10,000)
  * Charitable Contributions
  * Medical Expenses (above 7.5% AGI threshold)
  * Casualty Losses (federally declared disasters only)""",
        "source": "https://www.irs.gov/instructions/i1040",
        "form": "1040",
        "topic": "deductions",
    },
    {
        "content": """Standard Deduction Amounts 2024:
- Single: $14,600
- Married Filing Jointly: $27,700
- Married Filing Separately: $13,850
- Head of Household: $21,900
- Qualifying Widow(er): $27,700""",
        "source": "https://www.irs.gov/individuals/income-tax",
        "form": "1040",
        "topic": "standard_deduction",
    },
    {
        "content": """2024 Federal Income Tax Brackets:

Single:
- 10%: $0 - $11,600
- 12%: $11,601 - $47,150
- 22%: $47,151 - $100,525
- 24%: $100,526 - $191,950
- 32%: $191,951 - $243,725
- 35%: $243,726 - $609,350
- 37%: Over $609,350

Married Filing Jointly:
- 10%: $0 - $23,200
- 12%: $23,201 - $94,300
- 22%: $94,301 - $201,050
- 24%: $201,051 - $383,900
- 32%: $383,901 - $487,450
- 35%: $487,451 - $731,200
- 37%: Over $731,200""",
        "source": "https://www.irs.gov/income-tax/individual-income-tax-brackets",
        "form": "1040",
        "topic": "tax_brackets",
    },
    {
        "content": """Itemized Deductions (Schedule A):
- Mortgage Interest: Interest paid on loans secured by qualified residences
- State/Local Taxes: Sales and income taxes (combined cap $10,000)
- Charitable Gifts: Cash and property donations to qualified organizations
- Medical Expenses: Expenses exceeding 7.5% of AGI
- Home Office: Business use of home (must be regular, exclusive business use)
- Casualty Losses: Only for federally declared disasters
- Gambling: Losses up to amount of winnings (Schedule A)""",
        "source": "https://www.irs.gov/publications/p478",
        "form": "1040",
        "topic": "itemized_deductions",
    },
    {
        "content": """Child Tax Credit 2024:
- Amount: Up to $2,000 per qualifying child under 17
- Fully Refundable: For 2024, credit is fully refundable as American Opportunity Tax Credit
- Qualifying Child: Must be under 17, dependent on your return, U.S. citizen
- Phase-out: Begins at $200,000 (single) or $400,000 (married)""",
        "source": "https://www.irs.gov/credits-deductions/individual-credits",
        "form": "1040",
        "topic": "credits",
    },
    {
        "content": """Earned Income Tax Credit (EITC) 2024:
- Maximum Credit: $7,430 (3 children), $6,604 (2 children), $3,981 (1 child), $743 (no children)
- Income Limits 2024:
  * With 3+ children: $63,399 (single), $68,547 (married)
  * With 2 children: $60,074 (single), $65,222 (married)
  * With 1 child: $58,400 (single), $63,548 (married)
  * No children: $20,350 (single), $21,410 (married)
- Must have earned income (wages, self-employment)
- Must have valid SSN""",
        "source": "https://www.irs.gov/credits-deductions/earned-income-tax-credit-eitc",
        "form": "1040",
        "topic": "eitc",
    },
    {
        "content": """2024 Filing Requirements:

Single/Filer Separately/Head of Household:
File if ANY of these:
- Gross income >= $13,850
- Unearned income >= $1,250
- Social Security/RRA benefits >= $25,000
- Dependent filing threshold: $5,000

Married Filing Jointly/Separately:
File if ANY of these:
- Gross income >= $27,700
- Unearned income >= $1,250
- Social Security/RRA benefits >= $25,000
- Dependent filing threshold: $13,850""",
        "source": "https://www.irs.gov/income-taxes/who-must-file",
        "form": "1040",
        "topic": "filing_requirements",
    },
    {
        "content": """Schedule C - Profit or Loss From Business:
Report income from:
- Sole proprietorships
- Partnerships (individual's share)
- Independent contractors
- Gig economy work
- Freelance services

Required:
- Business name/location
- Industry code
- Gross receipts less returns/allowances
- Cost of goods sold
- Expenses (rent, supplies, utilities, home office)
- Net profit/loss flows to Form 1040

Self-Employment Tax: Calculate on Schedule SE (15.3% on net earnings)""",
        "source": "https://www.irs.gov/pub/irs-pdf/f1040instr.pdf",
        "form": "1040",
        "topic": "self_employment",
    },
    {
        "content": """Home Office Deduction Requirements:
1. Regular and exclusive use for business
2. Principal place of business
3. Used for administrative activities
4. Not used in connection with rental property only

Deduction Options:
- Simplified Method: $5 per square foot (max 300 sq ft = $1,500)
- Regular Method: Actual expenses (rent/mortgage, utilities, insurance, repairs) prorated by business-use percentage""",
        "source": "https://www.irs.gov/charity-charitable/contribution-of-non-cash-items",
        "form": "1040",
        "topic": "home_office",
    },
    {
        "content": """Common Tax Forms:

Form 1040: Individual Income Tax Return
Form 1040-SR: Seniors Tax Return
Form 1099-INT: Interest Income
Form 1099-DIV: Dividends
Form W-2: Wage and Tax Statement
Form 1099-NEC: Nonemployee Compensation
Form 1099-MISC: Miscellaneous Income
Schedule A: Itemized Deductions
Schedule C: Business Income/Loss
Schedule E: Rental Income
Schedule SE: Self-Employment Tax
Schedule AICPA: Agricultural Income""",
        "source": "https://www.irs.gov/forms-pubs/about-forms",
        "form": "forms",
        "topic": "forms",
    },
    {
        "content": """Deductible Business Expenses (Schedule C):
- Advertising
- Car (business miles only)
- Commissions
- Contracts and agreements
- Depreciable property
- Education and training
- Entertainment (generally not deductible)
- Filing fees
- Home office
- Insurance
- Interest (business)
- Legal and professional fees
- Office supplies
- Payroll and employment taxes
- Rent/lease of business property
- Repairs and maintenance
- Retirement plan contributions
- Safety equipment
- Travel and meal (50% deduction)
- Tools
- Utilities
- Wages and salaries (if employer)
- Writing fees""",
        "source": "https://www.irs.gov/businesses/small-businesses-self-employed/deductible-business-expenses",
        "form": "1040",
        "topic": "business_expenses",
    },
    {
        "content": """Tax Penalties:

Failure to File: 5% per month (max 25%)
Failure to Pay: 0.5% per month (max 25%)
Underpayment of Estimated Tax: 20% (if underpaid by $1,000+)
Late Form Penalty: Varies by form type
Fraud Penalty: 75% of underpayment (fraudulent return)
Neglect Penalty: 20% of underpayment
Accuracy-Related Penalty: 20% (negligence/substantial understatement)""",
        "source": "https://www.irs.gov/penalties",
        "form": "1040",
        "topic": "penalties",
    },
    {
        "content": """Amended Return (Form 1040-X):
Use when you need to:
- Claim a refund missed on original return
- Report additional income
- Claim deductions not taken
- Correct Social Security Number
- Change filing status
- Correct math errors

Processing time: Typically 3 months
Where to file: https://www.irs.gov/amending-form-1040-x
Required: Original return copy, corrected amounts, explanation of changes""",
        "source": "https://www.irs.gov/instructions/i1040x",
        "form": "1040-x",
        "topic": "amended_return",
    },
    {
        "content": """Roth IRA Contribution Limits 2024:

Maximum Contribution: $7,000 (under 50), $8,000 (50+)
Income Limits 2024:
- Phase-out begins: $146,000 (single), $218,000 (married filing jointly)
- Complete phase-out: $166,000 (single), $228,000 (married)

Contributions are made with after-tax dollars, so qualified distributions are tax-free.
Must be under 50 (50+ can contribute via backdoor Roth)""",
        "source": "https://www.irs.gov/retirement-plans/retirement-plan-limits",
        "form": "1040",
        "topic": "retirement",
    },
    {
        "content": """General Tax FAQ:

Q: Can I file my taxes early?
A: Yes, early filing is encouraged. Many states allow up to 30 days early.

Q: What documents do I need?
A: W-2, 1099s, bank statements, receipts for deductions, records of investment sales.

Q: Can I claim my child's dependents?
A: Only if they lived with you for more than half the year and meet support tests.

Q: What if I can't pay my taxes?
A: Set up an installment agreement with IRS at IRS.gov or call 1-800-829-1040.

Q: Where can I file?
A: File electronically (free for income under $72,500), mail to IRS processing center, or use commercial software.

Q: What is the statute of limitations?
A: IRS generally 3 years from filing date to audit. 6 years if you claim loss. Unlimited if fraud.""",
        "source": "https://www.irs.gov/help",
        "form": "1040",
        "topic": "faq",
    },
    {
        "content": """IRS Tax Forms by Category:

Business:
- Schedule C (Profit/Loss from Business)
- Schedule F (Farming)
- Schedule E (Supplemental Income)
- Schedule SE (Self-Employment Tax)

Deductions:
- Schedule A (Itemized Deductions)
- Schedule EIC (Earned Income Credit)

Credits:
- Form 8863 (American Opportunity Credit)
- Form 8814 (Taxable Distributions to Children)
- Form 3800 (Credit for Other Taxes Paid)

Retirement:
- Form 8606 (Nondeductible IRAs)
- Form 5498 (IRA Contributions)

Employment:
- Form W-2 (Wages and Taxes Withheld)
- Form W-4 (Withholding Certificate)
- Form W-3 (Transmittal of W-2s)""",
        "source": "https://www.irs.gov/forms-pubs/about-forms",
        "form": "forms",
        "topic": "forms_by_category",
    },
    {
        "content": """Estimated Tax Payments (Form 1040-ES):

Required if:
- You expect to owe tax and won't receive W-2
- Self-employed with net earnings > $400
- Married filing separately with withholding < 100% of prior year tax
- Expecting refund under $1,000

Payment schedule (4 quarterly):
- Form 1040-ES (Estimated Tax for Individuals)
- Payment due: April 15, June 15, September 15, January 15

Underpayment penalty if:
- Tax owed exceeds 100% of prior year tax
- Or 110% of prior year tax if AGI > $150,000""",
        "source": "https://www.irs.gov/individuals/estimated-tax",
        "form": "1040-es",
        "topic": "estimated_taxes",
    },
]

TAX_KEYWORDS = [
    "tax", "deduction", "filing", "1040", "bracket", "form", "irs",
    "wage", "income", "credit", "refund", "payment", "amended", "return",
    "schedule", "expense", "mortgage", "charitable", "self-employed",
    "schedule c", "itemized", "standard", "eic", "eitc", "roth",
    "ira", "estate", "gift", "penalty", "deadline", "extension",
    "form 1099", "w2", "1040x", "estimated", "quarterly",
    "business", "sole proprietor", "s corporation", "partnership",
    "llc", "corporation", "separate", "joint", "head of household",
    "married", "dependent", "child tax", "child credit", "saver's credit",
]

TAX_PHRASES = [
    "filing status", "tax bracket", "itemized deduction", "standard deduction",
    "schedule c", "form 1040", "income tax", "federal tax", "state tax",
    "estimated tax", "quarterly payment", "amended return", "tax credit",
    "tax refund", "tax return", "business expense", "home office",
    "self employment", "solar panels", "mortgage interest", "charitable contribution",
    "charitable deduction", "business income", "net worth", "estate tax",
    "farming income", "rental property", "real estate", "investment",
    "capital gains", "investment income", "dividend", "stock", "retirement",
]
