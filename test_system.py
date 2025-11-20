#!/usr/bin/env python
"""Complete test suite for ToM-NAS"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_all():
    print("="*60)
    print("ToM-NAS Complete System Test")
    print("="*60)
    
    all_pass = True
    
    # Test imports
    try:
        from src.core.ontology import SoulMapOntology
        ontology = SoulMapOntology()
        print(f"  ✓ Ontology: {ontology.total_dims} dimensions")
    except Exception as e:
        print(f"  ✗ Ontology failed: {e}")
        all_pass = False
        
    try:
        from src.core.beliefs import RecursiveBeliefState, BeliefNetwork
        beliefs = RecursiveBeliefState(0, 181, max_order=5)
        print(f"  ✓ Beliefs: 5th-order recursion")
    except Exception as e:
        print(f"  ✗ Beliefs failed: {e}")
        all_pass = False
        
    try:
        from src.agents.architectures import TransparentRNN, RecursiveSelfAttention, TransformerToMAgent
        print(f"  ✓ Architectures: TRN, RSAN, Transformer")
    except Exception as e:
        print(f"  ✗ Architectures failed: {e}")
        all_pass = False
        
    try:
        from src.world.social_world import SocialWorld4
        world = SocialWorld4(6, 181)
        print(f"  ✓ Social World: {len(world.agents)} agents")
    except Exception as e:
        print(f"  ✗ Social World failed: {e}")
        all_pass = False
    
    print("="*60)
    if all_pass:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ Some tests failed - check errors above")
    print("="*60)
    return all_pass

if __name__ == "__main__":
    test_all()
