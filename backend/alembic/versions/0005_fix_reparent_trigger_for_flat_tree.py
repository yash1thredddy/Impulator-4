"""fix reparent trigger ordering for flat version trees

Revision ID: 0005
Revises: 0004
Create Date: 2026-03-31
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION reparent_compound_children()
        RETURNS TRIGGER AS $$
        DECLARE
            v_promoted_id UUID;
        BEGIN
            -- Only act if the deleted compound is a parent (root of version tree)
            IF OLD.parent_id IS NOT NULL THEN
                RETURN OLD;
            END IF;

            -- Find the next oldest child to promote (lowest version, then earliest processed)
            SELECT entry_id INTO v_promoted_id
            FROM compounds
            WHERE parent_id = OLD.entry_id
            ORDER BY version ASC, processed_at ASC
            LIMIT 1;

            -- No children, nothing to do
            IF v_promoted_id IS NULL THEN
                RETURN OLD;
            END IF;

            -- Promote the chosen child to parent status before reparenting siblings.
            -- This keeps the flat-tree trigger satisfied during the sibling updates.
            UPDATE compounds
            SET parent_id = NULL, version = 1
            WHERE entry_id = v_promoted_id;

            -- Reparent remaining children to the promoted root compound
            UPDATE compounds
            SET parent_id = v_promoted_id
            WHERE parent_id = OLD.entry_id
              AND entry_id != v_promoted_id;

            RETURN OLD;
        END;
        $$ LANGUAGE plpgsql
        """
    )


def downgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION reparent_compound_children()
        RETURNS TRIGGER AS $$
        DECLARE
            v_promoted_id UUID;
        BEGIN
            IF OLD.parent_id IS NOT NULL THEN
                RETURN OLD;
            END IF;

            SELECT entry_id INTO v_promoted_id
            FROM compounds
            WHERE parent_id = OLD.entry_id
            ORDER BY version ASC, processed_at ASC
            LIMIT 1;

            IF v_promoted_id IS NULL THEN
                RETURN OLD;
            END IF;

            UPDATE compounds
            SET parent_id = v_promoted_id
            WHERE parent_id = OLD.entry_id
              AND entry_id != v_promoted_id;

            UPDATE compounds
            SET parent_id = NULL, version = 1
            WHERE entry_id = v_promoted_id;

            RETURN OLD;
        END;
        $$ LANGUAGE plpgsql
        """
    )
