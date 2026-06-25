import streamlit as st


def metric_card(label: str, value: str, delta: str = "", color: str = "blue") -> str:
    delta_html = ""
    if delta:
        cls = "delta-up" if delta.startswith("+") else "delta-down"
        delta_html = f'<div class="metric-delta {cls}">{delta}</div>'
    return f"""
    <div class="metric-card {color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def render_kpi_row(total: int, success_rate: float, pending: int, today: int) -> None:
    cols = st.columns(4)
    cards = [
        (cols[0], metric_card("Total Processed", str(total), color="blue")),
        (cols[1], metric_card("Success Rate", f"{success_rate:.1f}%",
                              "+2.3% vs last week" if success_rate > 0 else "", "green")),
        (cols[2], metric_card("Pending Review", str(pending),
                              f"{'⚠️ needs attention' if pending > 5 else ''}", "amber")),
        (cols[3], metric_card("Today's Volume", str(today), color="blue")),
    ]
    for col, html in cards:
        with col:
            st.markdown(html, unsafe_allow_html=True)
