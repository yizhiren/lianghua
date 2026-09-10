export function isStrategyScanning(submitting, latestRunStatus) {
  return submitting || latestRunStatus === "running";
}
