"""Domain wording for stages that write human-readable text (Stage 5
descriptions, Stage 7 narrative/prompt).

Read from the config's `domain` section so no stage hardcodes "employee" or
"attrition":

    domain:
      entity_noun: employee          # what one row is (default: "record")
      entity_noun_plural: employees  # default: entity_noun + "s"
      outcome_label: attrition       # human name of the outcome (default: the outcome column)

Defaults are resolved here, in one place, so a domain that says nothing gets
neutral wording rather than another domain's.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Vocabulary:
    entity: str
    entities: str
    outcome: str


def domain_vocabulary(domain_config: dict) -> Vocabulary:
    section = domain_config.get("domain", {})
    entity = section.get("entity_noun") or "record"
    return Vocabulary(
        entity=entity,
        entities=section.get("entity_noun_plural") or f"{entity}s",
        outcome=section.get("outcome_label") or domain_config["ingestion"]["outcome_column"],
    )
