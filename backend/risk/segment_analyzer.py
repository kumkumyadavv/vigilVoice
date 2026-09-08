def find_suspicious_segments(
    segments,
    threshold=0.70,
):
    suspicious = []

    for segment in segments:
        if segment["synthetic_probability"] >= threshold:
            suspicious.append(segment)

    return suspicious