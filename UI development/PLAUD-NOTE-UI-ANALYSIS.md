# PLAUD Note Web UI Analysis

> Reference for BlackBox V2 UI Redesign

---

## 1. Screenshots Analyzed

| Screenshot | View | Key Elements |
|------------|------|--------------|
| `142331.png` | **All Files (Home)** | Main list view, sidebar navigation, file table |
| `142735.png` | **Recording Detail** | Audio player, transcript area, sidebar list |
| `142924.png` | **Generation Modal** | Method selection dialog |
| `142938.png` | **Generation Progress** | Loading state with skeleton UI |

---

## 2. Design Philosophy

### Core Principles Observed

1. **Minimalism First** - No gradients, no shadows, no visual noise
2. **Content Density** - Efficient use of space, more data visible
3. **Functional Color** - Purple accent only for primary actions
4. **Typography-Driven** - Text hierarchy creates structure, not decorations
5. **Whitespace as Design** - Generous padding, clean separation

### Color Palette

| Element | Color | Hex (Estimated) |
|---------|-------|-----------------|
| Background | Pure White | `#FFFFFF` |
| Sidebar BG | Light Gray | `#FAFAFA` or `#F5F5F5` |
| Text Primary | Dark Gray | `#1A1A1A` or `#333333` |
| Text Secondary | Medium Gray | `#666666` |
| Text Muted | Light Gray | `#999999` |
| Accent/Primary | Purple | `#7C3AED` or `#8B5CF6` |
| Accent Hover | Darker Purple | `#6D28D9` |
| Border/Divider | Very Light Gray | `#E5E5E5` or `#EEEEEE` |
| Success Badge | Green | `#10B981` |
| Warning/Progress | Amber/Orange | `#F59E0B` |
| Selected Item BG | Very Light Purple | `#F3F0FF` |

---

## 3. Layout Structure

### Global Layout (3-Column)

```
┌─────────────────────────────────────────────────────────────────┐
│ HEADER BAR (minimal, logo left, user right)                     │
├──────────────┬──────────────────────────────────────────────────┤
│              │                                                   │
│   SIDEBAR    │              MAIN CONTENT AREA                   │
│   (Fixed)    │              (Scrollable)                        │
│   ~240px     │                                                   │
│              │                                                   │
│  - Nav       │  - Page Title                                    │
│  - Files     │  - Content (Table/Detail)                        │
│  - Filters   │                                                   │
│              │                                                   │
├──────────────┴──────────────────────────────────────────────────┤
│ FOOTER (minimal - plan info, upgrade button)                    │
└─────────────────────────────────────────────────────────────────┘
```

### Sidebar Structure (Left Panel)

```
┌──────────────────────┐
│ PLAUD (Logo)         │
│ ▼ Agneya (User)      │
├──────────────────────┤
│ [+ Add audio]        │  ← Primary CTA (Purple button)
├──────────────────────┤
│ 🔍 Search            │  ← Navigation items
│ 🏠 Home              │
│ 💬 Ask Plaud         │
│ 📋 Template Community│
│ 🔭 Explore           │
├──────────────────────┤
│ 📁 All files (85)    │  ← File categories (selected = purple)
│ 📝 Untitled (85)     │
│ 🗑️ Trash (0)         │
├──────────────────────┤
│ ▼ Folders            │  ← Collapsible section
├──────────────────────┤
│ Comes from           │  ← Filter section
│   Note - Note mode   │
│   Note - Call mode   │
│   Import             │
├──────────────────────┤
│                      │
│ Starter        ▸     │  ← Plan indicator
│ 300 mins left  ⓘ     │
│ [Go Unlimited]       │  ← Upgrade CTA
└──────────────────────┘
```

---

## 4. Component Breakdown

### 4.1 All Files View (Home Page)

#### Table Design

```
┌─────────────────────────────────────────────────────────────────┐
│ All files                                          Sort: Date ▼ │
├─────────────────────────────────────┬──────────┬────────────┬───┤
│ Name                                │ Duration │ Date created│ ⋮│
├─────────────────────────────────────┼──────────┼────────────┼───┤
│ 2025-12-10 16:11:21                │ 49m 30s  │ 2025-12-10 │ ⋮ │
│ ────────────────────────────────────┼──────────┼────────────┼───│
│ 12-09 Design Review Meeting: Off...│ 1h 0m 34s│ 2025-12-09 │ ⋮ │
│ ◎ Generated                         │          │ 21:06:46   │   │
├─────────────────────────────────────┼──────────┼────────────┼───┤
│ 2025-12-04 22:29:40                │ 2m 48s   │ 2025-12-04 │ ⋮ │
└─────────────────────────────────────┴──────────┴────────────┴───┘
```

#### Table Characteristics

| Property | Value |
|----------|-------|
| Row Height | ~60-70px (two-line items) |
| Row Padding | 16px horizontal |
| Dividers | 1px light gray horizontal lines |
| Hover State | Subtle background highlight |
| Column Alignment | Name (left), Duration (left), Date (left) |
| Actions | Three-dot menu (⋮) appears on hover |
| Badge | "Generated" with green checkmark icon |

#### Row Content Structure

```
Primary Text:   Date/Time or Title (bold, dark)
Secondary Text: Full title if truncated (lighter, smaller)
Badge:          "Generated" indicator (green, small)
Duration:       Formatted as "Xh Ym Zs" or "Xm Ys"
Date:           Full timestamp "YYYY-MM-DD HH:MM:SS"
```

---

### 4.2 Recording Detail View

#### Layout (Two-Panel)

```
┌────────────────────────┬────────────────────────────────────────┐
│   RECORDINGS LIST      │          DETAIL PANEL                  │
│   (Scrollable)         │                                        │
│                        │  ┌──────────────────────────────────┐  │
│ ┌────────────────────┐ │  │  Transcript │ Summary            │  │
│ │ ▌2025-12-10 16:11  │ │  └──────────────────────────────────┘  │
│ │   49m 30s          │ │                                        │
│ │   ● Generating...  │ │  ┌──────────────────────────────────┐  │
│ └────────────────────┘ │  │     Audio Player Bar              │  │
│ ┌────────────────────┐ │  │  ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬   │  │
│ │ 12-09 Design Rev...│ │  │  00:00 / 49:30                    │  │
│ │   1h 0m 34s        │ │  │     ⏪  ▶️  ⏩    1x  ⬇️  ⛶       │  │
│ │   ✓ Generated      │ │  └──────────────────────────────────┘  │
│ └────────────────────┘ │                                        │
│ ┌────────────────────┐ │  ┌──────────────────────────────────┐  │
│ │ 2025-12-04 22:29   │ │  │                                  │  │
│ │   2m 48s           │ │  │   📄 Ready to generate           │  │
│ └────────────────────┘ │  │                                  │  │
│                        │  │   Transcript will appear here    │  │
│        ...             │  │   after generation               │  │
│                        │  │                                  │  │
│                        │  │   [    🔮 Generate    ]          │  │
│                        │  │                                  │  │
│                        │  └──────────────────────────────────┘  │
└────────────────────────┴────────────────────────────────────────┘
```

#### Recording List Item (Selected State)

```css
/* Selected item has purple left border */
.recording-item.selected {
    background: #F3F0FF;  /* Very light purple */
    border-left: 3px solid #7C3AED;  /* Purple accent */
}
```

#### Audio Player Elements

| Element | Description |
|---------|-------------|
| Progress Bar | Thin line at top, purple filled portion |
| Timestamp | `00:16 / 49:30` format, left-aligned |
| Play/Pause | Large centered button |
| Skip | Forward/back 10s buttons |
| Speed | Playback speed selector (1x, 1.5x, 2x) |
| Download | Download button icon |
| Fullscreen | Expand to fullscreen |

#### Tab Navigation

```
┌──────────────┬──────────────┐
│  Transcript  │   Summary    │  ← Two tabs, underline active
└──────────────┴──────────────┘
```

- Active tab: Purple underline, darker text
- Inactive tab: No underline, lighter text

---

### 4.3 Generation Method Modal

#### Modal Design

```
┌────────────────────────────────────────────────────┐
│ Select generation method                        ✕  │
├────────────────────────────────────────────────────┤
│                                                    │
│  ◉ Auto generation                        Beta    │
│     Plaud generates the best transcript and       │
│     summarizes for you — no setup needed     ✓    │
│                                                    │
│  ○ Custom generation                              │
│     [Description text here]                       │
│                                                    │
├────────────────────────────────────────────────────┤
│           [ Generate now ]                         │
└────────────────────────────────────────────────────┘
```

#### Modal Characteristics

| Property | Value |
|----------|-------|
| Width | ~450-500px |
| Border Radius | 12px |
| Shadow | Subtle drop shadow |
| Overlay | Semi-transparent dark background |
| Close Button | X in top-right corner |
| Selection | Radio buttons with descriptions |
| CTA Button | Full-width purple button |

---

### 4.4 Generation Progress State

#### Progress Indicator

```
┌──────────────────────────────────────────────────┐
│                                                  │
│              Generating...                       │
│                                                  │
│   This may take a few minutes. You can leave    │
│   this page and come back later.                │
│                                                  │
│   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░    │
│   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░    │
│   ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░    │
│                                                  │
└──────────────────────────────────────────────────┘
```

#### Skeleton Loading Pattern

- Animated shimmer effect
- Gray placeholder bars of varying lengths
- 3-4 placeholder lines visible

#### List Item Badge Change

```
Before: ✓ Generated (green)
During: ● Generating... (orange/amber, possibly animated)
```

---

## 5. Typography

### Font Stack

```css
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
             'Noto Sans', Helvetica, Arial, sans-serif;
```

### Type Scale

| Element | Size | Weight | Color |
|---------|------|--------|-------|
| Page Title | 24px | 600 (Semibold) | #1A1A1A |
| Section Header | 14px | 500 (Medium) | #666666 |
| Table Header | 13px | 500 (Medium) | #666666 |
| Body Text | 14px | 400 (Regular) | #333333 |
| Secondary Text | 13px | 400 (Regular) | #666666 |
| Muted Text | 12px | 400 (Regular) | #999999 |
| Badge Text | 11px | 500 (Medium) | Varies |
| Button Text | 14px | 500 (Medium) | White |

---

## 6. Interactive States

### Buttons

| State | Style |
|-------|-------|
| Default | Purple background (#7C3AED), white text |
| Hover | Slightly darker purple (#6D28D9) |
| Active/Pressed | Even darker, slight scale down |
| Disabled | Gray background, muted text |

### Table Rows

| State | Style |
|-------|-------|
| Default | White background |
| Hover | Very light gray background (#F9F9F9) |
| Selected | Light purple background (#F3F0FF), purple left border |

### Navigation Items

| State | Style |
|-------|-------|
| Default | Gray text, no background |
| Hover | Slightly darker text, light background |
| Active | Purple text, light purple background, possible icon fill |

---

## 7. Comparison: PLAUD vs Current BlackBox V2

| Aspect | PLAUD Note | BlackBox V2 | Change Required |
|--------|-----------|-------------|-----------------|
| **Background** | Pure white | Purple gradient | Remove gradient |
| **Layout** | Sidebar + Table | Cards grid | Switch to table layout |
| **Cards/Items** | Simple rows | Cards with shadows | Remove cards, use rows |
| **Colors** | Monochrome + purple accent | Multi-color gradients | Simplify palette |
| **Stats Display** | None on home | Gradient stat cards | Remove or simplify |
| **Navigation** | Sidebar with sections | Header buttons | Add sidebar |
| **File List** | Table with columns | Card grid | Convert to table |
| **Shadows** | None/minimal | Heavy drop shadows | Remove shadows |
| **Emojis** | None | Heavy use (📤🎙️📅) | Remove emojis |
| **Typography** | Clean, simple | Gradient text effects | Simplify |
| **Buttons** | Solid purple, simple | Gradient with glow | Flatten buttons |
| **Hover Effects** | Subtle | Lift with shadow increase | Reduce/remove |
| **Border Radius** | 8-12px | 12px | Keep similar |

---

## 8. Missing Screenshots Needed

To complete the analysis, I would benefit from seeing:

### High Priority

1. **Transcript View** - What does the actual transcript look like when generated?
   - Text formatting
   - Timestamps display
   - Speaker labels (if any)
   - Copy/export buttons

2. **Summary View** - The "Summary" tab content
   - Structure of AI summary
   - Section formatting
   - Action items display

3. **Ask Plaud (Chat)** - The Q&A interface
   - Input field design
   - Message bubbles
   - Citation display

### Medium Priority

4. **Search Results** - How search results appear
5. **Settings/Preferences** - Any settings pages
6. **Mobile/Responsive** - How it looks on smaller screens
7. **Empty States** - What shows when no files exist
8. **Error States** - Error message styling

### Low Priority

9. **Template Community** - The templates page
10. **Explore** - The explore/discover page
11. **Folder Management** - Creating/managing folders

---

## 9. Implementation Roadmap for BlackBox

### Phase 1: Structure Overhaul
- [ ] Remove gradient background, use white
- [ ] Add left sidebar navigation
- [ ] Convert project grid to table layout
- [ ] Remove stat cards from home

### Phase 2: Simplify Visual Design
- [ ] Remove all drop shadows
- [ ] Remove all gradient effects
- [ ] Remove emojis from buttons/headings
- [ ] Implement monochrome + purple accent palette
- [ ] Flatten all buttons

### Phase 3: Component Polish
- [ ] Design table row component
- [ ] Create sidebar navigation component
- [ ] Design audio player bar
- [ ] Create tab navigation component
- [ ] Design modal component

### Phase 4: States & Interactions
- [ ] Implement hover states (subtle)
- [ ] Add selected states with purple accent
- [ ] Create loading/skeleton states
- [ ] Design progress indicators

### Phase 5: Detail Views
- [ ] Redesign project detail page
- [ ] Add transcript/summary tabs
- [ ] Redesign chat interface
- [ ] Add audio player

---

## 10. Key Design Tokens (CSS Variables)

```css
:root {
    /* Colors */
    --color-bg-primary: #FFFFFF;
    --color-bg-secondary: #FAFAFA;
    --color-bg-selected: #F3F0FF;

    --color-text-primary: #1A1A1A;
    --color-text-secondary: #666666;
    --color-text-muted: #999999;

    --color-accent: #7C3AED;
    --color-accent-hover: #6D28D9;
    --color-accent-light: #F3F0FF;

    --color-border: #E5E5E5;
    --color-divider: #EEEEEE;

    --color-success: #10B981;
    --color-warning: #F59E0B;
    --color-error: #EF4444;

    /* Typography */
    --font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                   'Noto Sans', Helvetica, Arial, sans-serif;

    --font-size-xs: 11px;
    --font-size-sm: 13px;
    --font-size-base: 14px;
    --font-size-lg: 16px;
    --font-size-xl: 20px;
    --font-size-2xl: 24px;

    --font-weight-normal: 400;
    --font-weight-medium: 500;
    --font-weight-semibold: 600;

    /* Spacing */
    --spacing-xs: 4px;
    --spacing-sm: 8px;
    --spacing-md: 16px;
    --spacing-lg: 24px;
    --spacing-xl: 32px;

    /* Layout */
    --sidebar-width: 240px;
    --border-radius-sm: 6px;
    --border-radius-md: 8px;
    --border-radius-lg: 12px;

    /* Transitions */
    --transition-fast: 150ms ease;
    --transition-normal: 200ms ease;
}
```

---

## 11. Summary

PLAUD Note's web interface exemplifies **functional minimalism**:

1. **No decoration for decoration's sake** - Every visual element serves a purpose
2. **Content is king** - The audio files and transcripts are the focus, not the UI
3. **Purple accent strategically used** - Only for primary actions and selection states
4. **Table over cards** - More efficient for scanning many items
5. **Sidebar navigation** - Familiar pattern, keeps actions accessible
6. **Progressive disclosure** - Details revealed on click, not shown all at once

### The Core Transformation

```
FROM: Visually rich, gradient-heavy, card-based, emoji-decorated
  TO: Clean, white, table-based, typography-focused
```

This is a **significant visual simplification** that will make BlackBox feel more professional and less "template-y".

---

*Document created: 2026-01-28*
*Reference screenshots: 4 files in `/UI development/plaude note web screenshots/`*
