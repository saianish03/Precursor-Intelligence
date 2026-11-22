---
name: Precursor Intelligence
description: Monthly nursing-home risk scores where every claim resolves to its CMS record.
colors:
  navy: "#14284b"
  navy-2: "#1d3560"
  rail-text: "#d3dceb"
  ground: "#f6f7f5"
  paper: "#ffffff"
  ink: "#16191d"
  muted: "#55606e"
  rule: "#c9ced6"
  rule-soft: "#e5e8ec"
  link: "#1a5dab"
  link-tint: "#eaf1fa"
  risk: "#8a2b1f"
  lowers: "#4f6b8a"
  attn: "#8a5a00"
  attn-tint: "#fbefd6"
  mark: "#fff4c2"
  mark-edge: "#c99a00"
  sev-low: "#e3e6ea"
  survey: "#6b7686"
typography:
  display:
    fontFamily: "Public Sans Variable, Public Sans, system-ui, sans-serif"
    fontSize: "2.125rem"
    fontWeight: 700
    lineHeight: 1
    fontFeature: "\"tnum\""
  headline:
    fontFamily: "Public Sans Variable, Public Sans, system-ui, sans-serif"
    fontSize: "1.5rem"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.01em"
  title:
    fontFamily: "Public Sans Variable, Public Sans, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 700
    lineHeight: 1.5
  body:
    fontFamily: "Public Sans Variable, Public Sans, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.5
    fontFeature: "\"tnum\""
  label:
    fontFamily: "Public Sans Variable, Public Sans, system-ui, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 600
    lineHeight: 1.45
  identifier:
    fontFamily: "Roboto Mono, ui-monospace, monospace"
    fontSize: "0.75rem"
    fontWeight: 400
    lineHeight: 1.45
  prose:
    fontFamily: "Merriweather, Georgia, serif"
    fontWeight: 400
rounded:
  sm: "3px"
  md: "4px"
spacing:
  cell-y: "6px"
  cell-x: "8px"
  control-y: "6px"
  control-x: "10px"
  rail-x: "20px"
  page-x-mobile: "16px"
  page-x: "28px"
  header: "24px"
  section: "32px"
components:
  button-primary:
    backgroundColor: "{colors.navy}"
    textColor: "{colors.paper}"
    rounded: "{rounded.md}"
    padding: "6px 12px"
  button-secondary:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: "6px 12px"
  button-secondary-hover:
    backgroundColor: "{colors.ground}"
  input-field:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: "6px 10px"
  evidence-chip:
    backgroundColor: "{colors.link-tint}"
    textColor: "{colors.link}"
    typography: "{typography.identifier}"
    rounded: "{rounded.sm}"
    padding: "1px 6px"
  evidence-chip-marked:
    backgroundColor: "{colors.mark}"
    textColor: "{colors.ink}"
  severity-harm:
    backgroundColor: "{colors.risk}"
    textColor: "{colors.paper}"
    rounded: "{rounded.sm}"
    padding: "0 4px"
  severity-no-harm:
    backgroundColor: "{colors.sev-low}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "0 4px"
  flag-attention:
    backgroundColor: "{colors.attn-tint}"
    textColor: "{colors.attn}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "2px 8px"
  nav-item:
    textColor: "{colors.rail-text}"
    padding: "10px 20px"
  nav-item-active:
    backgroundColor: "{colors.navy-2}"
    textColor: "{colors.paper}"
---

# Design System: Precursor Intelligence

## Overview

**Creative North Star: "The Register"**

The app reads like a Federal Register or CFR page set for a working desk: a GPO-navy rail on the left, a cool
off-white ground, white work surfaces, and black hairline rules that do the job cards usually do. Every number on
screen sits next to the record it came from, and those records can be opened. Structure comes from rules, weight and
alignment, not from boxes, tints or shadows. The pages are dense and calm. Hierarchy is set with three type sizes plus
one display figure, the facility's probability.

Colour carries meaning, not decoration. Navy is the institution (rail, sign-in panel, the one filled button). Citation
blue is anything you can follow. Oxblood is risk and G to L severity only. Amber is a state that needs attention. Pale
yellow is a record the reader has marked. Everything else is ink, grey and rule.

The world rejects the KPI-card dashboard and the chatbot pane. Tables, definition lists, fixed-scale bars and the
36-month citation strip are the main forms.

**Key Characteristics:**
- Navy rail (232px) plus a white freshness line (month, model, scored date, data flags) above every page.
- Hairline rules as structure: an ink rule under section titles and table heads, soft rules between rows.
- Tabular figures everywhere, and one identifier notation (Roboto Mono) for CCNs, evidence IDs and run IDs.
- Fixed scales: a 10-step decile meter, a 0-to-1 probability scale and a 36-month axis that read the same on every facility.
- A marked record holds its pale-yellow mark until the reader dismisses it.

## Colors

An institutional, low-chroma palette: navy and ink carry the structure, and three narrow signal colours (blue, oxblood,
amber) are each kept to one meaning.

### Primary
- **GPO Navy** (navy): the rail, the sign-in identity panel, the filled search button and the form accent colour. It is the institution's colour, never a status colour.
- **Rail Navy, Raised** (navy-2): hover and active fill for rail navigation items.
- **Rail Type** (rail-text): inactive nav labels and the wordmark subtitle on navy.

### Secondary
- **Citation Blue** (link): links, text buttons (Sign out), the focus ring and the caret. Anything blue can be followed.
- **Citation Chip Ground** (link-tint): the pale blue background of an evidence chip at rest.

### Tertiary
- **Oxblood** (risk): the probability figure, G to L severity badges, harm marks on the citation strip, the "raises risk" driver bars and the decile ramp. Nothing else.
- **Slate Blue, Lowering** (lowers): bars for features that lower risk, so raise and lower differ in hue and in list.
- **Attention Amber** (attn) on **Amber Wash** (attn-tint): data flags such as "Gold data not verified" and load warnings.
- **Marked-Record Yellow** (mark) with **Mark Edge Gold** (mark-edge): the highlight on a marked record, the marked chip, the mark bar and text selection.

### Neutral
- **Ink** (ink): body text, section rules and table-head rules.
- **Desk Grey** (muted): notes, table heads, captions and secondary lines.
- **Survey Grey** (survey): non-harm survey marks on the citation strip. It is 4.6:1 on paper, so it is used for marks that must meet 3:1.
- **Cool Ground** (ground): the page background behind work surfaces, and the hover fill of secondary buttons.
- **Paper** (paper): the freshness line, panels, inputs and type on navy or oxblood.
- **Rule** (rule): control borders, the freshness line's lower edge and panel borders.
- **Soft Rule** (rule-soft): row separators, bar tracks, empty decile cells and skeletons.
- **No-Harm Grey** (sev-low): the A to F severity badge.

### Named Rules
**The Oxblood Means Risk Rule.** Oxblood is used only for the model's risk and G to L severity. A heading, a button or an error is never oxblood.

**The Blue Is a Link Rule.** Citation blue marks only something you can follow or activate: a link, a chip, a text button, the focus ring. Never decoration.

**The Decile Ramp Rule.** Decile colour mixes oxblood into paper in OKLCH, 100% for the riskiest tenth down to 10% (never below 8%). The same ramp is used on every screen.

## Typography

**Display and Body Font:** Public Sans Variable (with Public Sans, system-ui)
**Label/Mono Font:** Roboto Mono (with ui-monospace), identifiers only
**Prose Font (reserved):** Merriweather (with Georgia)

**Character:** One sturdy government grotesque does all the UI and data work with tabular figures. Mono is set aside
for codes the reader may copy. Merriweather is loaded and tokenised but not set on any element yet: it is reserved for
agent prose and quoted regulation text, which arrive with the Investigation brief column.

### Hierarchy
- **Display** (700, 2.125rem, line-height 1): the facility's probability, in oxblood. One per page.
- **Headline** (700, 1.5rem, 1.2, -0.01em): page titles and the facility name.
- **Title** (700, 0.875rem, 1.5): section titles, over an ink rule. Rank comes from weight and the rule, not size.
- **Body** (400, 0.875rem, 1.5): everything else. Page intros are capped at 56rem.
- **Label** (600, 0.8125rem, 1.45): table heads (muted), form labels, freshness values and flags. Most tables are set at this size.
- **Identifier** (400 or 500, Roboto Mono, 0.75rem in chips and ID columns, 0.92em inline): CCNs, evidence IDs, gold run IDs and request IDs.

### Named Rules
**The Three Sizes Rule.** The app runs on 0.8125rem, 0.875rem and 1.5rem. Rank comes from weight (400/600/700), colour (ink or muted) and a rule. The display figure is the only exception.

**The Tabular Figures Rule.** `font-variant-numeric: tabular-nums` is set on the body and never turned off, so columns of numbers line up.

**The One Notation Rule.** An identifier is always mono, and evidence IDs wrap only after their colons.

## Layout

The shell is a two-column grid from 768px: a 232px sticky navy rail, then the content column. Below 768px the rail
collapses into a top bar with a menu toggle. The freshness line spans the content column (16px side padding on
mobile, 28px from 768px), and main content uses the same side padding with 24px top and bottom.

Pages use a 12-column grid from 1024px with 32px gaps (for example 8/4 and 6/6 on This month, 7/5 on Model health).
The facility profile is a header (name block on the left, probability block on the right, ink rule beneath), then from
1024px a fluid evidence column and a 400px sticky side column. The direction called for a 440px brief column; the
build ships 400px. Sections stack with 32px gaps, page headers sit 24px above content, and table cells use 6px by 8px
padding. Spacing follows Tailwind's 4px step.

Breakpoints: 640px (decile labels shown), 768px (rail appears), 1024px (multi-column pages), 1280px (evidence-ID column in tables).

## Elevation & Depth

Flat. Depth comes from the navy rail against the ground, white surfaces on the cool ground, and rules. There is one
shadow, and only on the floating mark bar, the one element that floats above the page.

### Shadow Vocabulary
- **Overlay** (`box-shadow: 0 6px 24px -6px rgb(20 40 75 / 0.25)`): the fixed mark bar. Nothing at rest gets a shadow.

### Named Rules
**The Rules, Not Cards Rule.** Sections are separated by an ink rule under their title and by whitespace, not by bordered or shadowed containers. Borders are kept for controls, panels for error and empty states, and the mark.

## Shapes

Almost square. Controls, panels and the mark bar use a 4px radius. Chips, badges, flags and skeletons use 3px. Only
the citation-strip survey marks and their legend dots are circles. Bars and decile cells are square. Lines are 1px
hairlines. Outlines are 2px, for focus (citation blue) and for a marked record (mark-edge gold).

## Components

### Buttons
Plain and few. One filled button per form at most.
- **Shape:** 4px radius.
- **Primary:** navy fill, paper icon or text, navy border. Used today on the search submit, joined to its input.
- **Secondary:** paper fill, rule border, 600-weight label, 6px by 12px. Hover fills with the ground colour. Used for "Try again".
- **Text button:** citation-blue 600-weight text that underlines on hover (Sign out, Clear).
- **Focus:** a 2px citation-blue outline, offset 2px, on every interactive element.

### Chips
- **Evidence chip:** mono 0.75rem in citation blue on the pale blue ground, 3px radius, 1px by 6px. It underlines on hover. Activating it marks the record: the chip turns mark yellow with a gold outline, and the record scrolls into view.
- **Severity badge:** a bold letter on a 3px tile at least 24px wide, oxblood with paper type for G to L and no-harm grey with ink type for A to F. Its title states the class, so colour is never the only signal.
- **Data flag:** 600-weight amber type on the amber wash, 3px radius, linking to the evidence behind it.

### Cards / Containers
- None for structure. Error and empty states are paper panels with a 4px radius, a rule border and 16px padding. Empty states use a dashed border and centred text.

### Inputs / Fields
- **Style:** paper fill, rule border, 4px radius, 6px by 10px, body type. Labels sit above, in 600 weight at label size.
- **Focus:** the global citation-blue ring. Native checkboxes and selects take the navy accent colour.

### Navigation
- **Rail:** wordmark (bold paper name, rail-text subtitle) above a vertical list of 500-weight items, 10px by 20px. Hover and active use the raised navy fill with paper type. On mobile the list sits behind a menu toggle and closes on navigation.
- **Freshness line:** a white strip with a rule beneath. Label-size muted text, with values in 600-weight ink, then flags, then the session on the right.

### The Citation Rail (signature)
Any record on the page (a score, a driver row, a citation, a penalty, a chain) carries its evidence ID. Activating a
chip or a strip mark highlights that record in mark yellow with a 2px gold outline, scrolls it to the centre, focuses
it, and opens the mark bar ("Marked `id`" with Clear). The mark stays until Escape or Clear. The colour change takes
200ms and is disabled when reduced motion is set.

### Fixed-Scale Marks
- **Decile meter:** ten 6px by 12px cells, 1px apart. The facility's tenth is filled from the decile ramp, the rest are soft rule, and risk grows to the right. The text label is always present (visually from 640px, otherwise for screen readers only).
- **Bar:** an 8px track in soft rule with a fill on a fixed 0-to-max scale. The value is always shown as text too.
- **Citation strip:** a 56px band over an ink baseline, with a fixed 36-month axis and year ticks in soft rule. Harm surveys sit in an upper row as 14px oxblood dots, other surveys in a lower row as 12px survey-grey dots, each with a 2px paper ring. Marks scale up on hover or focus.

## Do's and Don'ts

### Do:
- **Do** show the data month, model version and caveat next to every score: the freshness line on every page, and the "About this score" block on the facility.
- **Do** attach an evidence chip or a record anchor to every claim, so the citation rail can resolve it.
- **Do** separate sections with an ink rule under a bold 0.875rem title, and table rows with soft rules.
- **Do** keep the decile meter, the 0-to-1 bars and the 36-month axis fixed, so any two facilities compare at a glance.
- **Do** pair every colour signal with text: severity letters, decile labels, bar values as text, flag wording.
- **Do** set every identifier in Roboto Mono.

### Don't:
- **Don't** use oxblood for anything but model risk and G to L severity.
- **Don't** build pages from KPI cards or shadowed containers. Rules and whitespace carry the structure.
- **Don't** add a fourth UI type size. Rank comes from weight, colour and rule.
- **Don't** let a mark clear on blur or hover-out. It holds until the reader dismisses it.
- **Don't** set agent prose or regulation quotes in Public Sans once they ship. Merriweather is reserved for them, and only for them.
- **Don't** use amber for decoration. It is only for states that need attention.
