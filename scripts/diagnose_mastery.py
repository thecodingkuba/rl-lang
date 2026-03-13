"""
Diagnostic script: trace true vs belief mastery over an episode with
VocabDrill-only to understand why mastery doesn't grow.

Usage: python scripts/diagnose_mastery.py  (from project root)
"""

import numpy as np
from src.config.env_config import EnvConfig
from src.env.env import LanguageTutoringEnv


def main():
    config = EnvConfig()
    env = LanguageTutoringEnv(config)

    # Run one episode with VocabDrill only (action 0)
    obs, info = env.reset(seed=42)
    K = env.K

    true_curves = [info["true_mastery"].copy()]
    belief_curves = [info["belief_mastery"].copy()]

    for step in range(config.episode_length):
        obs, reward, term, trunc, info = env.step(0)  # VocabDrill
        true_curves.append(info["true_mastery"].copy())
        belief_curves.append(info["belief_mastery"].copy())
        if term or trunc:
            break

    true_curves = np.array(true_curves)
    belief_curves = np.array(belief_curves)

    # Summary
    print("=" * 60)
    print("DIAGNOSIS: Why Mastery Doesn't Grow")
    print("=" * 60)
    print(f"\nEpisode length: {len(true_curves) - 1} steps, VocabDrill only")
    print(f"Skills: {K} total (5 grammar, 5 vocab)")

    print("\n--- Mean over all skills ---")
    true_mean = true_curves.mean(axis=1)
    belief_mean = belief_curves.mean(axis=1)
    print(f"True mastery:  start={true_mean[0]:.4f}  end={true_mean[-1]:.4f}  Δ={true_mean[-1]-true_mean[0]:.4f}")
    print(f"Belief mastery: start={belief_mean[0]:.4f}  end={belief_mean[-1]:.4f}  Δ={belief_mean[-1]-belief_mean[0]:.4f}")

    # Per-skill breakdown (vocab vs grammar)
    grammar_idx = env.skill_map.grammar_indices
    vocab_idx = env.skill_map.vocab_indices

    true_grammar = true_curves[:, grammar_idx].mean(axis=1)
    true_vocab = true_curves[:, vocab_idx].mean(axis=1)
    belief_grammar = belief_curves[:, grammar_idx].mean(axis=1)
    belief_vocab = belief_curves[:, vocab_idx].mean(axis=1)

    print("\n--- By domain (VocabDrill never touches grammar) ---")
    print(f"True  grammar: start={true_grammar[0]:.4f}  end={true_grammar[-1]:.4f}  Δ={true_grammar[-1]-true_grammar[0]:.4f}")
    print(f"True  vocab:   start={true_vocab[0]:.4f}  end={true_vocab[-1]:.4f}  Δ={true_vocab[-1]-true_vocab[0]:.4f}")
    print(f"Belief grammar: start={belief_grammar[0]:.4f}  end={belief_grammar[-1]:.4f}  (never updated)")
    print(f"Belief vocab:   start={belief_vocab[0]:.4f}  end={belief_vocab[-1]:.4f}")

    # Root cause check
    print("\n--- Root cause ---")
    print("1. Belief for unpracticed skills: never updated (no decay in env)")
    print("   → grammar beliefs stay at init_mastery forever")
    print("2. VocabDrill sets difficulty = mastery → P(correct) = 0.5")
    print("   → E[update] = alpha * (0.5 - 0.5) = 0 for practiced skill")
    print("   → no expected growth from prediction-error at optimal challenge")
    print("3. True mastery: 9 skills decay every step, 1 gets E[update]=0")
    print("   → decay dominates, true mean drops (grammar drops most)")

    print("\n--- Expected from theory ---")
    decay_500 = (1 - config.decay_rate) ** 500
    print(f"Unpracticed skill after 500 decays: 0.25 * {1-config.decay_rate}^500 = {0.25 * decay_500:.4f}")
    env.close()


if __name__ == "__main__":
    main()
