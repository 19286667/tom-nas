"""
Institutional Framework for Computational Sociology

Institutions provide the social context within which researcher agents operate.
Each institution has:
- A mission and values (affecting what research is prioritized)
- Resources (compute, funding, data access)
- Reputation and influence
- Relationships with other institutions (collaboration, competition)

This creates realistic selective pressures that mirror how actual
scientific and technological progress emerges through institutional dynamics.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from collections import defaultdict
import uuid

from src.config import get_logger
from src.config.constants import SOUL_MAP_DIMS

logger = get_logger(__name__)


class InstitutionType(Enum):
    """Types of institutions with different incentive structures."""
    RESEARCH_LAB = "research_lab"           # Pure research, publication-focused
    CORPORATE_RD = "corporate_rd"           # Applied research, profit-focused
    GOVERNMENT_AGENCY = "government_agency" # Policy-focused, public good
    UNIVERSITY = "university"               # Education + research hybrid
    NONPROFIT = "nonprofit"                 # Mission-driven research
    STARTUP = "startup"                     # Innovation + growth focused


@dataclass
class InstitutionalValues:
    """
    Value profile that shapes what research an institution prioritizes.
    These create diverse selective pressures across the ecosystem.
    """
    # Research priorities (0-1 scale)
    theoretical_rigor: float = 0.5      # Pure vs applied
    practical_impact: float = 0.5       # Real-world application
    novelty: float = 0.5                # New discoveries vs refinement
    reproducibility: float = 0.5        # Verification emphasis
    collaboration: float = 0.5          # Open vs proprietary
    speed: float = 0.5                  # Fast publication vs thorough
    safety: float = 0.5                 # Risk tolerance
    transparency: float = 0.5          # Open science commitment

    def to_tensor(self) -> torch.Tensor:
        """Convert values to tensor for neural processing."""
        return torch.tensor([
            self.theoretical_rigor,
            self.practical_impact,
            self.novelty,
            self.reproducibility,
            self.collaboration,
            self.speed,
            self.safety,
            self.transparency,
        ])


@dataclass
class InstitutionalResources:
    """Resources available to an institution."""
    compute_budget: float = 100.0       # Computational resources
    funding: float = 100.0              # Financial resources
    data_access: float = 50.0           # Access to datasets
    talent_pool: int = 10               # Number of researcher slots
    infrastructure: float = 50.0        # Labs, equipment, etc.

    def consume(self, compute: float = 0, funding: float = 0) -> bool:
        """Consume resources, return False if insufficient."""
        if compute > self.compute_budget or funding > self.funding:
            return False
        self.compute_budget -= compute
        self.funding -= funding
        return True


@dataclass
class Institution:
    """
    Base class for all institutional actors.

    An institution provides context and resources for researcher agents,
    shaping their incentives and capabilities.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Unnamed Institution"
    institution_type: InstitutionType = InstitutionType.RESEARCH_LAB
    values: InstitutionalValues = field(default_factory=InstitutionalValues)
    resources: InstitutionalResources = field(default_factory=InstitutionalResources)

    # Relationships
    collaborators: Set[str] = field(default_factory=set)
    competitors: Set[str] = field(default_factory=set)

    # Track record
    publications: List[str] = field(default_factory=list)
    reputation: float = 50.0  # 0-100 scale
    influence: float = 50.0   # Policy/market influence

    # Research focus areas
    focus_areas: List[str] = field(default_factory=lambda: ["general"])

    def evaluate_research_proposal(self, proposal: Dict[str, Any]) -> float:
        """
        Score a research proposal based on institutional values.

        Returns a fitness score (0-1) indicating alignment with institution.
        """
        score = 0.0
        weights = self.values.to_tensor()

        # Score different aspects
        if "theoretical_contribution" in proposal:
            score += weights[0] * proposal["theoretical_contribution"]
        if "practical_impact" in proposal:
            score += weights[1] * proposal["practical_impact"]
        if "novelty_score" in proposal:
            score += weights[2] * proposal["novelty_score"]
        if "reproducibility" in proposal:
            score += weights[3] * proposal["reproducibility"]

        return float(score / weights.sum())

    def fund_research(self, cost: float) -> bool:
        """Attempt to fund a research project."""
        return self.resources.consume(funding=cost)

    def allocate_compute(self, amount: float) -> bool:
        """Allocate computational resources."""
        return self.resources.consume(compute=amount)

    def publish(self, publication_id: str, impact: float):
        """Record a publication and update reputation."""
        self.publications.append(publication_id)
        self.reputation = min(100, self.reputation + impact * 0.1)
        logger.info(f"{self.name} published {publication_id}, reputation: {self.reputation:.1f}")

    def collaborate_with(self, other_id: str):
        """Establish collaboration with another institution."""
        self.collaborators.add(other_id)
        if other_id in self.competitors:
            self.competitors.remove(other_id)

    def compete_with(self, other_id: str):
        """Mark another institution as competitor."""
        self.competitors.add(other_id)
        if other_id in self.collaborators:
            self.collaborators.remove(other_id)


class ResearchLab(Institution):
    """
    Academic research laboratory focused on fundamental discoveries.

    Prioritizes theoretical rigor, novelty, and publication.
    """
    def __init__(self, name: str = "Research Lab", **kwargs):
        values = InstitutionalValues(
            theoretical_rigor=0.9,
            practical_impact=0.3,
            novelty=0.8,
            reproducibility=0.7,
            collaboration=0.8,
            speed=0.4,
            safety=0.6,
            transparency=0.9,
        )
        super().__init__(
            name=name,
            institution_type=InstitutionType.RESEARCH_LAB,
            values=values,
            **kwargs
        )


class CorporateRD(Institution):
    """
    Corporate R&D department focused on profitable applications.

    Prioritizes practical impact, speed, and proprietary advantages.
    """
    def __init__(self, name: str = "Corporate R&D", **kwargs):
        values = InstitutionalValues(
            theoretical_rigor=0.4,
            practical_impact=0.9,
            novelty=0.6,
            reproducibility=0.5,
            collaboration=0.3,  # More proprietary
            speed=0.8,
            safety=0.5,
            transparency=0.3,
        )
        resources = InstitutionalResources(
            compute_budget=500.0,  # More resources
            funding=500.0,
            data_access=80.0,
            talent_pool=20,
            infrastructure=100.0,
        )
        super().__init__(
            name=name,
            institution_type=InstitutionType.CORPORATE_RD,
            values=values,
            resources=resources,
            **kwargs
        )


class GovernmentAgency(Institution):
    """
    Government research agency focused on public good and policy.

    Prioritizes safety, transparency, and broad impact.
    """
    def __init__(self, name: str = "Government Agency", **kwargs):
        values = InstitutionalValues(
            theoretical_rigor=0.6,
            practical_impact=0.7,
            novelty=0.5,
            reproducibility=0.8,
            collaboration=0.7,
            speed=0.3,  # Slower, more careful
            safety=0.9,
            transparency=0.8,
        )
        resources = InstitutionalResources(
            compute_budget=300.0,
            funding=400.0,
            data_access=90.0,  # Government data access
            talent_pool=15,
            infrastructure=80.0,
        )
        super().__init__(
            name=name,
            institution_type=InstitutionType.GOVERNMENT_AGENCY,
            values=values,
            resources=resources,
            **kwargs
        )


class InstitutionalNetwork:
    """
    Network of institutions that interact, collaborate, and compete.

    This creates the macro-level dynamics of a research ecosystem.
    """

    def __init__(self):
        self.institutions: Dict[str, Institution] = {}
        self.collaboration_graph: Dict[str, Set[str]] = defaultdict(set)
        self.competition_graph: Dict[str, Set[str]] = defaultdict(set)
        self.funding_flows: Dict[Tuple[str, str], float] = {}

    def add_institution(self, institution: Institution):
        """Add an institution to the network."""
        self.institutions[institution.id] = institution
        logger.info(f"Added {institution.name} ({institution.institution_type.value}) to network")

    def create_default_ecosystem(self) -> 'InstitutionalNetwork':
        """
        Create a realistic research ecosystem with diverse institutions.

        This provides the social environment for researcher agents.
        """
        # Academic labs
        self.add_institution(ResearchLab(name="Institute for Advanced Study"))
        self.add_institution(ResearchLab(name="Cognitive Science Lab"))
        self.add_institution(ResearchLab(name="AI Safety Research Center"))

        # Corporate R&D
        self.add_institution(CorporateRD(name="TechCorp AI Division"))
        self.add_institution(CorporateRD(name="DataSystems Research"))

        # Government
        self.add_institution(GovernmentAgency(name="National AI Initiative"))
        self.add_institution(GovernmentAgency(name="Defense Research Agency"))

        # Establish some initial relationships
        institutions = list(self.institutions.values())
        for i, inst1 in enumerate(institutions):
            for inst2 in institutions[i+1:]:
                # Similar types tend to compete
                if inst1.institution_type == inst2.institution_type:
                    inst1.compete_with(inst2.id)
                    inst2.compete_with(inst1.id)
                # Different types may collaborate
                elif inst1.values.collaboration > 0.5 and inst2.values.collaboration > 0.5:
                    inst1.collaborate_with(inst2.id)
                    inst2.collaborate_with(inst1.id)

        return self

    def get_ecosystem_state(self) -> Dict[str, Any]:
        """Get current state of the institutional ecosystem."""
        return {
            "num_institutions": len(self.institutions),
            "total_compute": sum(i.resources.compute_budget for i in self.institutions.values()),
            "total_funding": sum(i.resources.funding for i in self.institutions.values()),
            "avg_reputation": sum(i.reputation for i in self.institutions.values()) / max(1, len(self.institutions)),
            "collaboration_edges": sum(len(i.collaborators) for i in self.institutions.values()) // 2,
            "competition_edges": sum(len(i.competitors) for i in self.institutions.values()) // 2,
        }

    def step(self):
        """
        Advance the ecosystem by one time step.

        - Regenerate some resources
        - Update relationships based on recent interactions
        - Adjust reputations based on publication impact
        """
        for inst in self.institutions.values():
            # Partial resource regeneration
            inst.resources.compute_budget = min(
                inst.resources.compute_budget + 10,
                500 if inst.institution_type == InstitutionType.CORPORATE_RD else 200
            )
            inst.resources.funding = min(
                inst.resources.funding + 20,
                1000 if inst.institution_type == InstitutionType.CORPORATE_RD else 400
            )

            # Reputation decay (need to keep publishing)
            inst.reputation = max(0, inst.reputation - 0.5)
