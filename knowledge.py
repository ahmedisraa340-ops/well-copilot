"""Petroleum engineering knowledge base used by the diagnosis agent.
Edit this file with YOUR domain expertise: it is the team's competitive advantage."""

CAUSES = {
    "water_breakthrough": {
        "label": "Water breakthrough / coning",
        "signature": "Water cut rises steadily over weeks-months, oil declines faster than the underlying trend, WHP roughly stable.",
        "checks": ["Compare with offset injector rates and voidage replacement", "Run production logging (PLT) to locate water entry",
                   "Review choke/drawdown history (coning risk)", "Consider water shut-off or rate management"],
    },
    "artificial_lift_failure": {
        "label": "Artificial lift / ESP / pump problem",
        "signature": "Abrupt sustained rate drop, wellhead pressure falls, water cut unchanged, no choke change.",
        "checks": ["Check motor amps, intake pressure and frequency trends", "Verify pump efficiency and run-life vs design",
                   "Plan diagnostics (sonolog / dynamometer) and workover screening", "Check power supply and VSD alarms"],
    },
    "choke_change": {
        "label": "Choke / operational rate change",
        "signature": "Rate step change coincides with a choke change and WHP moves in the opposite direction.",
        "checks": ["Confirm choke change was intentional (operations log)", "Re-run well test at new choke",
                   "Check erosion/plugging if choke size unchanged but rate fell"],
    },
    "reservoir_depletion": {
        "label": "Reservoir pressure depletion / normal decline",
        "signature": "Smooth gradual decline, GOR slowly rising, no discrete events.",
        "checks": ["Compare with Arps forecast", "Review static pressure survey", "Evaluate pressure support / infill potential"],
    },
    "gas_breakthrough": {
        "label": "Gas breakthrough / gas cap coning",
        "signature": "GOR rises sharply, oil rate declines, water cut stable.",
        "checks": ["Review GOR trend vs solution GOR", "Consider reducing drawdown", "Check gas cap expansion / offset wells"],
    },
    "wellbore_issue": {
        "label": "Wellbore restriction (scale / wax / sand / hydrates)",
        "signature": "Rate and tubing pressure decline gradually together, may recover after intervention.",
        "checks": ["Compare tubing head vs flowing pressure trends", "Review chemical treatment programme", "Consider coiled-tubing cleanout"],
    },
    "shut_in": {
        "label": "Planned/unplanned shut-in",
        "signature": "On-stream hours drop to ~0.",
        "checks": ["Check ESD/ops log for cause", "Verify restart rate recovers to pre-shut-in level"],
    },
}
