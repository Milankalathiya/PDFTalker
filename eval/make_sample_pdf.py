"""Generate eval/sample_policy.pdf, a fictional insurance policy used by evaluation.py.

The facts sit on known pages so retrieval can be scored by page, and a few answers
depend on exact codes (e.g. HS-EX-04) where keyword search beats pure embeddings.
Run: python eval/make_sample_pdf.py  (needs reportlab, listed in requirements-eval.txt)
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGES = [
    ("HomeShield Home Insurance — Policy Wording", [
        "This is a fictional sample document created for testing PDF Talker. Harbourline Mutual and "
        "HomeShield are invented names. Policy reference series: HS-2026.",
        "Welcome to HomeShield. This policy wording, together with your schedule, forms your contract "
        "of insurance with Harbourline Mutual. Please read it carefully and keep it somewhere safe.",
        "Cooling-off period. You may cancel this policy within 21 days of receiving your documents, or "
        "from the start date if later, and receive a full refund of premium provided no claim has been made.",
        "Contacting us. Our claims helpline is open 24 hours a day, 7 days a week. For policy changes, "
        "our customer service team is available Monday to Friday, 8am to 6pm, and Saturday, 9am to 1pm.",
    ]),
    ("Definitions", [
        "Home: the private dwelling at the address shown in your schedule, including its garages and "
        "outbuildings, used solely for domestic purposes.",
        "Unoccupied: when your home has not been lived in by you or a member of your family for more "
        "than 45 consecutive days. Living in means sleeping there on a regular basis.",
        "Valuables: jewellery, watches, precious metals, furs, works of art and collections of stamps "
        "or coins.",
        "Excess: the first part of any claim that you must pay. Unless a different amount is stated "
        "for a specific cover, the standard excess is £250 for each claim.",
        "Sum insured: the most we will pay for each section, as shown in your schedule.",
    ]),
    ("Section A — Buildings", [
        "We cover your buildings against loss or damage caused by fire, lightning, explosion, storm, "
        "flood, theft or attempted theft, vandalism, falling trees, and escape of water or oil from "
        "any fixed heating or plumbing installation.",
        "Escape of water excess. For loss or damage caused by escape of water, such as a burst pipe, "
        "the excess is £500 instead of the standard excess.",
        "Subsidence. We cover damage caused by subsidence, heave or landslip of the site on which "
        "your buildings stand. The subsidence excess is £1,000.",
        "Alternative accommodation. If your home cannot be lived in following insured damage, we will "
        "pay reasonable costs of comparable alternative accommodation for you and your pets, up to 20% "
        "of the buildings sum insured.",
    ]),
    ("Section B — Contents", [
        "We cover your contents against the same causes listed in Section A while they are inside "
        "your home.",
        "Single article limit. The most we will pay for any one item, pair or set is £2,000 unless the "
        "item is individually listed in your schedule.",
        "Valuables limit. The most we will pay in total for all valuables is £10,000 per claim.",
        "Contents in the garden. Garden furniture, plants and equipment are covered in the open up to "
        "£1,000 in total.",
        "Students' belongings. Belongings of a family member who is in full-time education and living "
        "away from home, for example in university halls, are covered up to £3,000.",
        "Accidental damage to contents is not included as standard. It is available as an optional "
        "add-on for an additional premium.",
    ]),
    ("General Exclusions", [
        "This policy does not cover the following. Each exclusion carries a reference code that is "
        "quoted in claim decisions.",
        "HS-EX-01 Wear and tear: loss or damage caused by wear and tear, gradual deterioration, or "
        "lack of maintenance.",
        "HS-EX-02 Pre-existing damage: damage that happened before the policy started.",
        "HS-EX-03 Pollution: loss or damage caused by pollution or contamination, unless caused by a "
        "sudden and unforeseen incident.",
        "HS-EX-04 Vermin: loss or damage caused by insects, rats, mice, squirrels or other vermin.",
        "HS-EX-05 Unoccupied homes: escape of water, theft and vandalism are not covered while your "
        "home is unoccupied.",
        "HS-EX-06 Cyber: loss caused by computer viruses or the malicious use of electronic systems.",
        "HS-EX-07 Deliberate acts: loss or damage caused deliberately by you or anyone living with you.",
    ]),
    ("Making a Claim", [
        "Tell us as soon as possible, and in any case within 30 days, about any incident that may lead "
        "to a claim. Late notification may reduce the amount we pay.",
        "Theft and vandalism must be reported to the police within 48 hours, and you must give us the "
        "crime reference number.",
        "Once we have agreed the value of your claim, we aim to pay within 10 working days.",
        "Complaints. If you are unhappy, contact our complaints team first. If we have not resolved "
        "your complaint within 8 weeks, or you disagree with our final response, you may refer it to "
        "the independent ombudsman service free of charge.",
    ]),
    ("Premiums and Cancellation", [
        "You can pay annually or in 12 monthly instalments. Monthly instalments are interest free.",
        "Cancellation after the cooling-off period. You may cancel at any time. We will refund the "
        "premium for the unused period on a pro rata basis, less an administration fee of £35, "
        "provided no claim has been made in the current period of insurance.",
        "No-claims discount. For each consecutive claim-free year your premium is reduced, up to a "
        "maximum discount of 30% after 5 claim-free years.",
    ]),
    # Pages 8+ add realistic bulk and look-alike figures (other limits, excesses and time
    # periods) so retrieval has to pick the right passage, not just any passage with a number.
    ("Section C — Personal Possessions (optional)", [
        "If shown in your schedule, we cover personal possessions that you or your family normally "
        "wear or carry, anywhere in the world, against loss, theft or accidental damage.",
        "Single article limit away from home. The most we will pay for any one personal possession is "
        "£1,500, and for pedal cycles £750 each, unless individually listed in your schedule.",
        "Personal possessions excess. An excess of £100 applies to each claim under this section.",
        "Personal money and cards. We cover cash and bank notes up to £300, and fraudulent use of "
        "lost or stolen credit cards up to £1,000, provided the loss is reported to the card issuer "
        "within 24 hours of discovery.",
        "Worldwide cover is limited to 60 days in any one period of insurance for trips outside the "
        "United Kingdom.",
    ]),
    ("Section D — Legal Expenses (optional)", [
        "If shown in your schedule, we pay professional fees and costs up to £50,000 for any one claim "
        "for legal disputes about your employment, your home, consumer contracts and personal injury.",
        "Reasonable prospects. We will only pay costs where it is more likely than not that you will "
        "recover damages or obtain a legal remedy. Our legal advisers will assess this before we agree "
        "to fund a case.",
        "Time limits. The dispute must arise after the start of the policy, and you must report it to "
        "us within 180 days of the date you first became aware of it.",
        "Legal helpline. A confidential legal advice helpline is available 24 hours a day for any "
        "personal legal problem, whether or not it leads to a claim. No excess applies to this section.",
    ]),
    ("Section E — Home Emergency (optional)", [
        "If shown in your schedule, we will send an approved contractor to deal with an emergency at "
        "your home, such as a failure of your main heating system, a blocked drain, a roof leak or a "
        "loss of domestic power supply.",
        "Call-out limit. We pay up to £1,000 including VAT for each emergency, covering call-out, "
        "labour, parts and materials needed to make a temporary repair.",
        "Boiler age. Boilers more than 15 years old are covered for a temporary repair only, and "
        "replacement parts are not included.",
        "Waiting period. No claims can be made under this section during the first 14 days after it "
        "is added, unless it was added at renewal with no break in cover.",
    ]),
    ("How We Settle Claims", [
        "Buildings. We will pay the cost of repairing or rebuilding the damaged part of your buildings "
        "to the same standard and size, provided they were in a good state of repair. If repairs are "
        "not carried out, we will pay the reduction in market value caused by the damage.",
        "Contents. We will repair or replace damaged items with new items of the same quality where "
        "possible. Clothing and household linen are settled after a deduction for wear and tear.",
        "Matching items. We will not pay to replace undamaged parts of a set, suite or carpet that "
        "form a pair or set with a damaged item, unless they cannot be matched, in which case we pay "
        "up to 50% of the cost of the undamaged items.",
        "Underinsurance. If your buildings or contents sum insured is less than the full rebuild or "
        "replacement value, we may reduce any claim payment in proportion to the amount of "
        "underinsurance.",
    ]),
    ("Policy Conditions", [
        "Reasonable care. You must take all reasonable steps to prevent loss or damage and keep your "
        "home in a good state of repair.",
        "Security. When your home is left empty, all external doors must be locked and any alarm set. "
        "If the schedule shows an alarm condition, the alarm must be maintained under an annual "
        "service contract.",
        "Changes you must tell us about. You must tell us within 14 days if you change address, if "
        "building work over £20,000 is planned, or if the home will be let to tenants or used for "
        "business purposes.",
        "Other insurance. If there is any other insurance covering the same loss, we will only pay our "
        "share of the claim.",
        "Heating in winter. Between 1 November and 31 March, if your home is left empty for more than "
        "72 hours, you must either keep the heating on at a minimum of 12 degrees Celsius or drain the "
        "water system.",
    ]),
    ("Fraud and Data Protection", [
        "Fraud. If any claim is fraudulent or exaggerated, we will not pay it, we may cancel the policy "
        "from the date of the fraudulent act, and we may keep any premium you have paid and recover "
        "costs from you.",
        "How we use your information. We use your personal information to provide your policy, handle "
        "claims, prevent fraud and meet our legal obligations. We may share information with fraud "
        "prevention agencies and claims databases.",
        "Keeping your data. We keep policy and claims records for 7 years after the end of our "
        "relationship with you, unless the law requires a longer period.",
        "Your rights. You can ask for a copy of the information we hold about you, ask us to correct "
        "it, or object to certain uses. Requests are answered within one month.",
    ]),
    ("Renewal", [
        "We will write to you at least 21 days before your renewal date with your new premium and any "
        "changes to your cover.",
        "Automatic renewal. If you pay by continuous payment authority or direct debit, your policy will "
        "renew automatically unless you tell us otherwise. You can opt out of automatic renewal at any "
        "time by contacting customer service.",
        "Index linking. Your buildings sum insured is adjusted at each renewal in line with a rebuild "
        "cost index, and your contents sum insured in line with a consumer prices index. Index linking "
        "never reduces your sum insured.",
    ]),
    ("Glossary of Terms Used in Claim Decisions", [
        "Accidental damage: sudden, unexpected and visible damage that has not been caused on purpose.",
        "Escape of water: water escaping from a fixed water tank, pipe, appliance or heating system, "
        "including a burst pipe or a leaking washing machine hose.",
        "Storm: strong winds of at least 55 miles per hour, sometimes with heavy rain, snow or hail.",
        "Flood: water from outside the home rising above ground level, such as from a river, the sea or "
        "overwhelmed drains.",
        "Period of insurance: the length of time covered by this policy, as shown in the schedule, "
        "normally 12 months.",
        "Heave: upward movement of the ground beneath the buildings, often caused by swelling clay soil.",
    ]),
]


def main():
    out = Path(__file__).with_name("sample_policy.pdf")
    styles = getSampleStyleSheet()
    story = []
    for i, (heading, paragraphs) in enumerate(PAGES):
        story.append(Paragraph(heading, styles["Title" if i == 0 else "Heading1"]))
        for text in paragraphs:
            story += [Paragraph(text, styles["BodyText"]), Spacer(1, 8)]
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(out), pagesize=A4, title="HomeShield Policy Wording (fictional sample)").build(story)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
