-- =============================================================================
-- Supabase schema for Pronoun Annotation Cloud App
-- Run this in Supabase SQL Editor to create the required tables.
-- =============================================================================

-- Table: annotations
-- Stores pronoun annotations and no_pronoun markers per annotator.
CREATE TABLE IF NOT EXISTS annotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    annotator_id TEXT NOT NULL,
    poem_id TEXT NOT NULL,
    sentence_id INTEGER NOT NULL,
    no_pronoun BOOLEAN DEFAULT FALSE,
    pronoun TEXT,
    lemma TEXT,
    person TEXT,
    number TEXT,
    is_dropped BOOLEAN,
    position INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_annotations_annotator ON annotations(annotator_id);
CREATE INDEX IF NOT EXISTS idx_annotations_annotator_poem ON annotations(annotator_id, poem_id);
CREATE INDEX IF NOT EXISTS idx_annotations_annotator_poem_sent ON annotations(annotator_id, poem_id, sentence_id);

-- Table: poem_perspectives
-- Stores poem-level perspective judgments per annotator.
CREATE TABLE IF NOT EXISTS poem_perspectives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    annotator_id TEXT NOT NULL,
    poem_id TEXT NOT NULL,
    perspective_primary TEXT NOT NULL,
    perspective_secondary TEXT DEFAULT '',
    author TEXT,
    poem_date TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(annotator_id, poem_id)
);

CREATE INDEX IF NOT EXISTS idx_poem_perspectives_annotator ON poem_perspectives(annotator_id);

-- RLS is disabled by default. Data isolation is enforced in application code
-- by filtering all queries with annotator_id. For stricter security, enable RLS
-- and add policies that restrict access by annotator_id.
