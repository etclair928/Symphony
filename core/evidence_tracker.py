# =================================================================
# MODULE: core/evidence_tracker.py
# VERSION: 5.6.1
# DESCRIPTION: Reference counting for spectral evidence.
#
# FIXES: #4 - Evidence lifecycle safety
# =================================================================

from typing import Dict, Set, List, Optional
from dataclasses import dataclass, field
from threading import RLock

from core.feature_bundle import EvidenceType


@dataclass
class EvidenceLeaseRecord:
    """Track who has borrowed which evidence."""
    stage_name: str
    evidence_type: EvidenceType
    acquired_at: float
    expires_at: Optional[float] = None


class EvidenceTracker:
    """
    Reference-counted evidence lifecycle manager.

    Prevents premature release by tracking all borrowers.
    """

    def __init__(self):
        self._leases: Dict[EvidenceType, EvidenceLeaseRecord] = {}
        self._ref_counts: Dict[EvidenceType, int] = {}
        self._lock = RLock()

    def acquire(self, evidence_type: EvidenceType, stage_name: str, ttl: Optional[float] = None) -> bool:
        """Acquire a reference to evidence. Returns False if already leased by different stage."""
        with self._lock:
            # Check if already leased
            if evidence_type in self._leases:
                existing = self._leases[evidence_type]
                if existing.stage_name != stage_name:
                    return False

            # Update ref count
            self._ref_counts[evidence_type] = self._ref_counts.get(evidence_type, 0) + 1

            # Create or update lease
            import time
            self._leases[evidence_type] = EvidenceLeaseRecord(
                stage_name=stage_name,
                evidence_type=evidence_type,
                acquired_at=time.time(),
                expires_at=time.time() + ttl if ttl else None
            )
            return True

    def release(self, evidence_type: EvidenceType, stage_name: str) -> bool:
        """Release a reference. Returns True if last reference and can be freed."""
        with self._lock:
            # Check if caller owns the lease (or lease expired)
            lease = self._leases.get(evidence_type)
            if lease:
                import time
                is_owner = lease.stage_name == stage_name
                is_expired = lease.expires_at is not None and time.time() > lease.expires_at
                if not (is_owner or is_expired):
                    return False

            # Decrement ref count
            current = self._ref_counts.get(evidence_type, 0)
            if current <= 0:
                return True  # Already zero, can free

            self._ref_counts[evidence_type] = current - 1
            can_free = self._ref_counts[evidence_type] <= 0

            if can_free and evidence_type in self._leases:
                del self._leases[evidence_type]
                del self._ref_counts[evidence_type]

            return can_free

    def get_ref_count(self, evidence_type: EvidenceType) -> int:
        """Get current reference count for debugging."""
        return self._ref_counts.get(evidence_type, 0)

    def get_all_leases(self) -> Dict[EvidenceType, str]:
        """Get all active leases (evidence -> owner stage)."""
        with self._lock:
            return {et: lease.stage_name for et, lease in self._leases.items()}

    def can_release_safely(self, evidence_type: EvidenceType, stage_name: str) -> bool:
        """Check if a stage can safely release evidence."""
        with self._lock:
            lease = self._leases.get(evidence_type)
            if not lease:
                return True
            return lease.stage_name == stage_name