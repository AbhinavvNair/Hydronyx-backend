# PowerShell script to test Advanced Groundwater API
# Run this with: .\test_api.ps1

Write-Host "Testing Advanced Groundwater API" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

$baseUrl = "http://localhost:8001"

# Test 1: Health Check
Write-Host "1. Health Check" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$baseUrl/api/health" -Method GET
    Write-Host "   Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "   Response: $($response.Content)" -ForegroundColor Gray
} catch {
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 2: Spatiotemporal Forecast
Write-Host "2. Spatiotemporal Forecast (Feature A)" -ForegroundColor Yellow
try {
    $body = @{
        state = "maharashtra"
        months_ahead = 12
        method = "gnn"
        include_uncertainty = $true
        n_samples = 50
    } | ConvertTo-Json

    $response = Invoke-WebRequest -Uri "$baseUrl/api/predict_spatiotemporal" `
        -Method POST `
        -ContentType "application/json" `
        -Body $body

    Write-Host "   Status: $($response.StatusCode)" -ForegroundColor Green
    $json = $response.Content | ConvertFrom-Json
    Write-Host "   State: $($json.state)" -ForegroundColor Gray
    Write-Host "   Forecast Horizon: $($json.forecast_horizon) months" -ForegroundColor Gray
    Write-Host "   Predictions: $($json.predictions.Count) data points" -ForegroundColor Gray
} catch {
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 3: Counterfactual Simulation
Write-Host "3. Counterfactual Simulation (Feature B)" -ForegroundColor Yellow
try {
    $body = @{
        state = "karnataka"
        months_ahead = 12
        interventions = @{
            pumping = -0.2
            recharge = 1.5
        }
        n_bootstrap = 100
    } | ConvertTo-Json

    $response = Invoke-WebRequest -Uri "$baseUrl/api/counterfactual" `
        -Method POST `
        -ContentType "application/json" `
        -Body $body

    Write-Host "   Status: $($response.StatusCode)" -ForegroundColor Green
    $json = $response.Content | ConvertFrom-Json
    Write-Host "   State: $($json.state)" -ForegroundColor Gray
    Write-Host "   Mean Treatment Effect: $($json.treatment_effect.mean_effect)" -ForegroundColor Gray
    Write-Host "   Final Effect: $($json.treatment_effect.final_effect)" -ForegroundColor Gray
} catch {
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 4: Recharge Site Optimization
Write-Host "4. Recharge Site Optimization (Feature E)" -ForegroundColor Yellow
try {
    $body = @{
        state = "tamil nadu"
        nl_query = "Find 10 high-impact sites with low cost"
        n_sites = 10
        n_candidates = 100
    } | ConvertTo-Json

    $response = Invoke-WebRequest -Uri "$baseUrl/api/recharge_sites" `
        -Method POST `
        -ContentType "application/json" `
        -Body $body

    Write-Host "   Status: $($response.StatusCode)" -ForegroundColor Green
    $json = $response.Content | ConvertFrom-Json
    Write-Host "   State: $($json.state)" -ForegroundColor Gray
    Write-Host "   Sites Selected: $($json.selected_sites.Count)" -ForegroundColor Gray
    Write-Host "   Objectives: $($json.objectives.Count)" -ForegroundColor Gray
    
    if ($json.selected_sites.Count -gt 0) {
        Write-Host "   Top Site Score: $($json.selected_sites[0].total_score)" -ForegroundColor Gray
    }
} catch {
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "All tests complete!" -ForegroundColor Green
Write-Host ""
Write-Host "For interactive testing, visit:" -ForegroundColor Cyan
Write-Host "  http://localhost:8001/docs" -ForegroundColor White
