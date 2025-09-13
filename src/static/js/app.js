/**
 * Ceramic Armor ML Prediction System - Frontend Application
 * 
 * This file contains all the JavaScript functionality for the web interface,
 * including form handling, API communication, result visualization, and user interactions.
 */

class CeramicArmorApp {
    constructor() {
        this.currentTab = 'prediction';
        this.predictionHistory = [];
        this.charts = {};
        
        // Initialize the application
        this.init();
    }

    /**
     * Initialize the application
     */
    init() {
        this.setupEventListeners();
        this.setupSliders();
        this.setupFileUpload();
        this.loadPresets();
        this.updateCompositionTotal();
    }

    /**
     * Setup all event listeners
     */
    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Form interactions
        document.getElementById('predict-btn').addEventListener('click', () => {
            this.makePrediction();
        });

        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetForm();
        });

        document.getElementById('load-preset-btn').addEventListener('click', () => {
            this.showPresetModal();
        });

        // Batch processing
        document.getElementById('process-batch-btn').addEventListener('click', () => {
            this.processBatch();
        });

        // Composition sliders
        document.querySelectorAll('#sic, #b4c, #al2o3, #wc, #tic').forEach(slider => {
            slider.addEventListener('input', () => {
                this.updateSliderValue(slider);
                this.updateCompositionTotal();
            });
        });

        // Porosity slider
        document.getElementById('porosity').addEventListener('input', (e) => {
            this.updateSliderValue(e.target);
        });

        // Form validation
        document.querySelectorAll('input, select').forEach(input => {
            input.addEventListener('change', () => {
                this.validateForm();
            });
        });
    }

    /**
     * Setup slider functionality
     */
    setupSliders() {
        document.querySelectorAll('input[type="range"]').forEach(slider => {
            this.updateSliderValue(slider);
        });
    }

    /**
     * Update slider value display
     */
    updateSliderValue(slider) {
        const valueSpan = slider.parentElement.querySelector('.slider-value');
        if (valueSpan) {
            let value = parseFloat(slider.value);
            let displayValue;
            
            if (slider.id === 'porosity') {
                displayValue = (value * 100).toFixed(1) + '%';
            } else if (['sic', 'b4c', 'al2o3', 'wc', 'tic'].includes(slider.id)) {
                displayValue = (value * 100).toFixed(0) + '%';
            } else {
                displayValue = value.toString();
            }
            
            valueSpan.textContent = displayValue;
        }
    }

    /**
     * Update composition total and validate
     */
    updateCompositionTotal() {
        const sic = parseFloat(document.getElementById('sic').value) || 0;
        const b4c = parseFloat(document.getElementById('b4c').value) || 0;
        const al2o3 = parseFloat(document.getElementById('al2o3').value) || 0;
        const wc = parseFloat(document.getElementById('wc').value) || 0;
        const tic = parseFloat(document.getElementById('tic').value) || 0;
        
        const total = sic + b4c + al2o3 + wc + tic;
        const totalElement = document.getElementById('total-composition');
        
        totalElement.textContent = (total * 100).toFixed(1) + '%';
        
        // Add validation styling
        const compositionTotalElement = totalElement.parentElement;
        if (total > 1.01) {
            compositionTotalElement.classList.add('invalid');
            totalElement.textContent += ' (Exceeds 100%)';
        } else if (total < 0.01) {
            compositionTotalElement.classList.add('invalid');
            totalElement.textContent += ' (Too low)';
        } else {
            compositionTotalElement.classList.remove('invalid');
        }
    }

    /**
     * Switch between tabs
     */
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(button => {
            button.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');

        this.currentTab = tabName;

        // Load specific tab content
        if (tabName === 'results') {
            this.loadResultsHistory();
        }
    }

    /**
     * Validate form inputs
     */
    validateForm() {
        const errors = [];
        
        // Validate composition
        const total = this.getCompositionTotal();
        if (total > 1.01) {
            errors.push('Total composition exceeds 100%');
        } else if (total < 0.01) {
            errors.push('Total composition must be at least 1%');
        }

        // Validate processing parameters
        const temperature = parseFloat(document.getElementById('temperature').value);
        if (temperature < 1200 || temperature > 2500) {
            errors.push('Temperature must be between 1200-2500°C');
        }

        const pressure = parseFloat(document.getElementById('pressure').value);
        if (pressure < 1 || pressure > 200) {
            errors.push('Pressure must be between 1-200 MPa');
        }

        // Enable/disable predict button
        const predictBtn = document.getElementById('predict-btn');
        predictBtn.disabled = errors.length > 0;

        return errors;
    }

    /**
     * Get current composition total
     */
    getCompositionTotal() {
        const sic = parseFloat(document.getElementById('sic').value) || 0;
        const b4c = parseFloat(document.getElementById('b4c').value) || 0;
        const al2o3 = parseFloat(document.getElementById('al2o3').value) || 0;
        const wc = parseFloat(document.getElementById('wc').value) || 0;
        const tic = parseFloat(document.getElementById('tic').value) || 0;
        
        return sic + b4c + al2o3 + wc + tic;
    }

    /**
     * Collect form data for API request
     */
    collectFormData() {
        return {
            composition: {
                SiC: parseFloat(document.getElementById('sic').value) || 0,
                B4C: parseFloat(document.getElementById('b4c').value) || 0,
                Al2O3: parseFloat(document.getElementById('al2o3').value) || 0,
                WC: parseFloat(document.getElementById('wc').value) || 0,
                TiC: parseFloat(document.getElementById('tic').value) || 0
            },
            processing: {
                sintering_temperature: parseFloat(document.getElementById('temperature').value),
                pressure: parseFloat(document.getElementById('pressure').value),
                grain_size: parseFloat(document.getElementById('grain-size').value),
                holding_time: parseFloat(document.getElementById('holding-time').value) || 120,
                heating_rate: parseFloat(document.getElementById('heating-rate').value) || 15,
                atmosphere: document.getElementById('atmosphere').value
            },
            microstructure: {
                porosity: parseFloat(document.getElementById('porosity').value),
                phase_distribution: document.getElementById('phase-distribution').value,
                interface_quality: document.getElementById('interface-quality').value,
                pore_size: parseFloat(document.getElementById('pore-size').value) || 1.0
            },
            include_uncertainty: document.getElementById('include-uncertainty').checked,
            include_feature_importance: document.getElementById('include-feature-importance').checked,
            prediction_type: document.querySelector('input[name="prediction-type"]:checked').value
        };
    }

    /**
     * Make prediction API call
     */
    async makePrediction() {
        const errors = this.validateForm();
        if (errors.length > 0) {
            this.showToast('Please fix form errors before predicting', 'error');
            return;
        }

        const formData = this.collectFormData();
        const predictionType = formData.prediction_type;
        
        this.showLoading(true);
        
        try {
            let results = {};
            
            // Make API calls based on prediction type
            if (predictionType === 'mechanical' || predictionType === 'both') {
                const mechanicalResponse = await this.callPredictionAPI('/api/v1/predict/mechanical', formData);
                results.mechanical = mechanicalResponse;
            }
            
            if (predictionType === 'ballistic' || predictionType === 'both') {
                const ballisticResponse = await this.callPredictionAPI('/api/v1/predict/ballistic', formData);
                results.ballistic = ballisticResponse;
            }
            
            // Display results
            this.displayResults(results, formData);
            
            // Add to history
            this.addToHistory(results, formData);
            
            this.showToast('Prediction completed successfully!', 'success');
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.showToast(`Prediction failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Call prediction API
     */
    async callPredictionAPI(endpoint, data) {
        try {
            const response = await axios.post(endpoint, data, {
                headers: {
                    'Content-Type': 'application/json'
                },
                timeout: 30000 // 30 second timeout
            });
            
            return response.data;
        } catch (error) {
            if (error.response) {
                // Server responded with error status
                const errorData = error.response.data;
                throw new Error(errorData.detail?.message || errorData.message || 'Server error');
            } else if (error.request) {
                // Request was made but no response received
                throw new Error('No response from server. Please check your connection.');
            } else {
                // Something else happened
                throw new Error(error.message || 'Unknown error occurred');
            }
        }
    }

    /**
     * Display prediction results
     */
    displayResults(results, formData) {
        const resultsContainer = document.getElementById('results-container');
        resultsContainer.innerHTML = '';

        // Create results HTML
        let resultsHTML = '<div class="prediction-results">';

        // Mechanical results
        if (results.mechanical) {
            resultsHTML += this.createMechanicalResultsHTML(results.mechanical);
        }

        // Ballistic results
        if (results.ballistic) {
            resultsHTML += this.createBallisticResultsHTML(results.ballistic);
        }

        // Feature importance
        if (formData.include_feature_importance && (results.mechanical || results.ballistic)) {
            const featureData = results.mechanical?.feature_importance || results.ballistic?.feature_importance;
            if (featureData && featureData.length > 0) {
                resultsHTML += this.createFeatureImportanceHTML(featureData);
            }
        }

        resultsHTML += '</div>';
        resultsContainer.innerHTML = resultsHTML;

        // Create charts if needed
        this.createResultCharts(results, formData);
    }

    /**
     * Create mechanical results HTML
     */
    createMechanicalResultsHTML(data) {
        const predictions = data.predictions;
        
        return `
            <div class="results-section-header">
                <h3><i class="fas fa-cog"></i> Mechanical Properties</h3>
            </div>
            <div class="property-cards">
                ${this.createPropertyCard('Fracture Toughness', predictions.fracture_toughness)}
                ${this.createPropertyCard('Vickers Hardness', predictions.vickers_hardness)}
                ${this.createPropertyCard('Density', predictions.density)}
                ${this.createPropertyCard('Elastic Modulus', predictions.elastic_modulus)}
            </div>
        `;
    }

    /**
     * Create ballistic results HTML
     */
    createBallisticResultsHTML(data) {
        const predictions = data.predictions;
        
        return `
            <div class="results-section-header">
                <h3><i class="fas fa-shield-alt"></i> Ballistic Properties</h3>
            </div>
            <div class="property-cards">
                ${this.createPropertyCard('V50 Velocity', predictions.v50_velocity)}
                ${this.createPropertyCard('Penetration Resistance', predictions.penetration_resistance)}
                ${this.createPropertyCard('Back-face Deformation', predictions.back_face_deformation)}
                ${this.createPropertyCard('Multi-hit Capability', predictions.multi_hit_capability)}
            </div>
        `;
    }

    /**
     * Create property card HTML
     */
    createPropertyCard(title, property) {
        const uncertaintyClass = `uncertainty-${property.prediction_quality || 'good'}`;
        const confidenceInterval = property.confidence_interval || [property.value, property.value];
        
        return `
            <div class="property-card">
                <h4>${title}</h4>
                <div class="property-value">${property.value.toFixed(3)}</div>
                <div class="property-unit">${property.unit}</div>
                <div class="confidence-interval">
                    CI: [${confidenceInterval[0].toFixed(3)}, ${confidenceInterval[1].toFixed(3)}]
                </div>
                <div class="uncertainty-badge ${uncertaintyClass}">
                    ${(property.uncertainty * 100).toFixed(1)}% uncertainty
                </div>
            </div>
        `;
    }

    /**
     * Create feature importance HTML
     */
    createFeatureImportanceHTML(featureData) {
        const topFeatures = featureData.slice(0, 10); // Show top 10 features
        
        let html = `
            <div class="feature-importance">
                <h4><i class="fas fa-chart-bar"></i> Feature Importance</h4>
        `;
        
        topFeatures.forEach(feature => {
            const percentage = (feature.importance * 100).toFixed(1);
            html += `
                <div class="feature-item">
                    <span class="feature-name">${feature.name}</span>
                    <div class="feature-bar">
                        <div class="feature-bar-fill" style="width: ${percentage}%"></div>
                    </div>
                    <span class="feature-value">${percentage}%</span>
                </div>
            `;
        });
        
        html += '</div>';
        return html;
    }

    /**
     * Create result charts
     */
    createResultCharts(results, formData) {
        // Clear existing charts
        Object.values(this.charts).forEach(chart => {
            if (chart) chart.destroy();
        });
        this.charts = {};

        // Create charts container if it doesn't exist
        let chartsContainer = document.getElementById('charts-container');
        if (!chartsContainer) {
            chartsContainer = document.createElement('div');
            chartsContainer.id = 'charts-container';
            chartsContainer.className = 'charts-container';
            document.getElementById('results-container').appendChild(chartsContainer);
        } else {
            chartsContainer.innerHTML = '';
        }

        // Create property comparison chart
        this.createPropertyComparisonChart(results, chartsContainer);

        // Create uncertainty chart if uncertainty is included
        if (formData.include_uncertainty) {
            this.createUncertaintyChart(results, chartsContainer);
        }

        // Create feature importance chart if included
        if (formData.include_feature_importance) {
            this.createFeatureImportanceChart(results, chartsContainer);
        }

        // Create confidence interval chart
        this.createConfidenceIntervalChart(results, chartsContainer);

        // Add download buttons
        this.addDownloadButtons(results, formData, chartsContainer);
    }

    /**
     * Create property comparison chart
     */
    createPropertyComparisonChart(results, container) {
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        chartContainer.innerHTML = `
            <h4><i class="fas fa-chart-bar"></i> Property Values Comparison</h4>
            <div class="chart-controls">
                <label>
                    <input type="radio" name="chart-type" value="radar" checked> Radar Chart
                </label>
                <label>
                    <input type="radio" name="chart-type" value="bar"> Bar Chart
                </label>
            </div>
            <canvas id="property-comparison-chart"></canvas>
        `;
        container.appendChild(chartContainer);

        const ctx = document.getElementById('property-comparison-chart').getContext('2d');
        
        // Collect all properties with their values and units
        const allProperties = [];
        
        if (results.mechanical) {
            Object.entries(results.mechanical.predictions).forEach(([key, prop]) => {
                allProperties.push({
                    name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                    value: prop.value,
                    unit: prop.unit,
                    type: 'Mechanical',
                    uncertainty: prop.uncertainty,
                    confidence_interval: prop.confidence_interval
                });
            });
        }
        
        if (results.ballistic) {
            Object.entries(results.ballistic.predictions).forEach(([key, prop]) => {
                allProperties.push({
                    name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                    value: prop.value,
                    unit: prop.unit,
                    type: 'Ballistic',
                    uncertainty: prop.uncertainty,
                    confidence_interval: prop.confidence_interval
                });
            });
        }

        // Normalize values for radar chart (0-100 scale)
        const normalizedData = allProperties.map(prop => {
            // Simple normalization - could be improved with domain knowledge
            return Math.min(100, Math.max(0, (prop.value / (prop.value + 1)) * 100));
        });

        const labels = allProperties.map(prop => prop.name);
        const mechanicalData = [];
        const ballisticData = [];
        
        allProperties.forEach((prop, index) => {
            if (prop.type === 'Mechanical') {
                mechanicalData.push(normalizedData[index]);
                ballisticData.push(null);
            } else {
                ballisticData.push(normalizedData[index]);
                mechanicalData.push(null);
            }
        });

        this.charts.propertyComparison = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Mechanical Properties',
                        data: mechanicalData,
                        backgroundColor: 'rgba(37, 99, 235, 0.2)',
                        borderColor: 'rgba(37, 99, 235, 1)',
                        borderWidth: 2,
                        pointBackgroundColor: 'rgba(37, 99, 235, 1)',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: 'rgba(37, 99, 235, 1)'
                    },
                    {
                        label: 'Ballistic Properties',
                        data: ballisticData,
                        backgroundColor: 'rgba(16, 185, 129, 0.2)',
                        borderColor: 'rgba(16, 185, 129, 1)',
                        borderWidth: 2,
                        pointBackgroundColor: 'rgba(16, 185, 129, 1)',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: 'rgba(16, 185, 129, 1)'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Property Values Comparison (Normalized 0-100 Scale)'
                    },
                    legend: {
                        position: 'top'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const prop = allProperties[context.dataIndex];
                                if (prop && context.parsed.r !== null) {
                                    return `${context.dataset.label}: ${prop.value.toFixed(3)} ${prop.unit} (±${(prop.uncertainty * 100).toFixed(1)}%)`;
                                }
                                return null;
                            }
                        }
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        pointLabels: {
                            font: {
                                size: 10
                            }
                        },
                        ticks: {
                            display: false
                        }
                    }
                },
                interaction: {
                    intersect: false
                }
            }
        });

        // Add chart type toggle functionality
        const radioButtons = chartContainer.querySelectorAll('input[name="chart-type"]');
        radioButtons.forEach(radio => {
            radio.addEventListener('change', () => {
                if (radio.value === 'bar') {
                    this.switchToBarChart(ctx, allProperties);
                } else {
                    this.switchToRadarChart(ctx, allProperties);
                }
            });
        });

        // Store properties for chart switching
        this.currentProperties = allProperties;
    }

    /**
     * Switch property chart to bar chart
     */
    switchToBarChart(ctx, properties) {
        if (this.charts.propertyComparison) {
            this.charts.propertyComparison.destroy();
        }

        const labels = properties.map(prop => prop.name);
        const values = properties.map(prop => prop.value);
        const colors = properties.map(prop => 
            prop.type === 'Mechanical' ? 'rgba(37, 99, 235, 0.6)' : 'rgba(16, 185, 129, 0.6)'
        );

        this.charts.propertyComparison = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Property Values',
                    data: values,
                    backgroundColor: colors,
                    borderColor: colors.map(color => color.replace('0.6', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Property Values (Actual Units)'
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const prop = properties[context.dataIndex];
                                return `${prop.value.toFixed(3)} ${prop.unit} (±${(prop.uncertainty * 100).toFixed(1)}%)`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Property Value'
                        }
                    },
                    x: {
                        ticks: {
                            maxRotation: 45
                        }
                    }
                }
            }
        });
    }

    /**
     * Switch property chart to radar chart
     */
    switchToRadarChart(ctx, properties) {
        // This will recreate the original radar chart
        this.createPropertyComparisonChart({ 
            mechanical: this.currentResults?.results?.mechanical, 
            ballistic: this.currentResults?.results?.ballistic 
        }, document.getElementById('charts-container'));
    }

    /**
     * Create uncertainty visualization chart
     */
    createUncertaintyChart(results, container) {
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        chartContainer.innerHTML = `
            <h4><i class="fas fa-exclamation-triangle"></i> Prediction Uncertainty Analysis</h4>
            <canvas id="uncertainty-chart"></canvas>
        `;
        container.appendChild(chartContainer);

        const ctx = document.getElementById('uncertainty-chart').getContext('2d');
        
        // Prepare data
        const labels = [];
        const uncertainties = [];
        const colors = [];
        
        // Collect data from results
        if (results.mechanical) {
            const props = results.mechanical.predictions;
            Object.entries(props).forEach(([key, prop]) => {
                labels.push(key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()));
                uncertainties.push(prop.uncertainty * 100);
                
                // Color code by uncertainty level
                const uncertainty = prop.uncertainty * 100;
                if (uncertainty < 5) {
                    colors.push('rgba(16, 185, 129, 0.8)'); // Green - excellent
                } else if (uncertainty < 10) {
                    colors.push('rgba(37, 99, 235, 0.8)'); // Blue - good
                } else if (uncertainty < 20) {
                    colors.push('rgba(245, 158, 11, 0.8)'); // Yellow - fair
                } else {
                    colors.push('rgba(239, 68, 68, 0.8)'); // Red - poor
                }
            });
        }
        
        if (results.ballistic) {
            const props = results.ballistic.predictions;
            Object.entries(props).forEach(([key, prop]) => {
                labels.push(key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()));
                uncertainties.push(prop.uncertainty * 100);
                
                const uncertainty = prop.uncertainty * 100;
                if (uncertainty < 5) {
                    colors.push('rgba(16, 185, 129, 0.8)');
                } else if (uncertainty < 10) {
                    colors.push('rgba(37, 99, 235, 0.8)');
                } else if (uncertainty < 20) {
                    colors.push('rgba(245, 158, 11, 0.8)');
                } else {
                    colors.push('rgba(239, 68, 68, 0.8)');
                }
            });
        }

        this.charts.uncertainty = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Uncertainty (%)',
                    data: uncertainties,
                    backgroundColor: colors,
                    borderColor: colors.map(color => color.replace('0.8', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Prediction Uncertainty by Property'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: Math.max(...uncertainties) * 1.2,
                        title: {
                            display: true,
                            text: 'Uncertainty (%)'
                        }
                    },
                    x: {
                        ticks: {
                            maxRotation: 45
                        }
                    }
                }
            }
        });
    }

    /**
     * Create feature importance chart
     */
    createFeatureImportanceChart(results, container) {
        const featureData = results.mechanical?.feature_importance || results.ballistic?.feature_importance;
        if (!featureData || featureData.length === 0) return;

        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        chartContainer.innerHTML = `
            <h4><i class="fas fa-chart-bar"></i> Feature Importance Analysis</h4>
            <canvas id="feature-importance-chart"></canvas>
        `;
        container.appendChild(chartContainer);

        const ctx = document.getElementById('feature-importance-chart').getContext('2d');
        
        // Get top 15 features for better insight
        const topFeatures = featureData.slice(0, 15);
        const labels = topFeatures.map(f => f.name.replace(/_/g, ' '));
        const importances = topFeatures.map(f => f.importance * 100);

        // Create gradient colors based on importance
        const colors = importances.map((importance, index) => {
            const intensity = importance / Math.max(...importances);
            return `rgba(16, 185, 129, ${0.3 + intensity * 0.7})`;
        });

        this.charts.featureImportance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Importance (%)',
                    data: importances,
                    backgroundColor: colors,
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Top 15 Most Important Features'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Importance (%)'
                        }
                    },
                    y: {
                        ticks: {
                            font: {
                                size: 10
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Create confidence interval visualization
     */
    createConfidenceIntervalChart(results, container) {
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        chartContainer.innerHTML = `
            <h4><i class="fas fa-chart-line"></i> Confidence Intervals</h4>
            <canvas id="confidence-interval-chart"></canvas>
        `;
        container.appendChild(chartContainer);

        const ctx = document.getElementById('confidence-interval-chart').getContext('2d');
        
        const labels = [];
        const values = [];
        const lowerBounds = [];
        const upperBounds = [];
        
        // Collect data from results
        if (results.mechanical) {
            Object.entries(results.mechanical.predictions).forEach(([key, prop]) => {
                labels.push(key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()));
                values.push(prop.value);
                lowerBounds.push(prop.confidence_interval[0]);
                upperBounds.push(prop.confidence_interval[1]);
            });
        }
        
        if (results.ballistic) {
            Object.entries(results.ballistic.predictions).forEach(([key, prop]) => {
                labels.push(key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()));
                values.push(prop.value);
                lowerBounds.push(prop.confidence_interval[0]);
                upperBounds.push(prop.confidence_interval[1]);
            });
        }

        this.charts.confidenceInterval = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Predicted Value',
                        data: values,
                        borderColor: 'rgba(37, 99, 235, 1)',
                        backgroundColor: 'rgba(37, 99, 235, 0.1)',
                        borderWidth: 3,
                        pointRadius: 6,
                        pointBackgroundColor: 'rgba(37, 99, 235, 1)',
                        tension: 0.1
                    },
                    {
                        label: 'Upper Bound (95% CI)',
                        data: upperBounds,
                        borderColor: 'rgba(239, 68, 68, 0.8)',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 3,
                        fill: false
                    },
                    {
                        label: 'Lower Bound (95% CI)',
                        data: lowerBounds,
                        borderColor: 'rgba(239, 68, 68, 0.8)',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 3,
                        fill: '+1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Prediction Values with 95% Confidence Intervals'
                    },
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            maxRotation: 45
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Property Value'
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    }

    /**
     * Add prediction to history
     */
    addToHistory(results, formData) {
        const historyItem = {
            id: Date.now(),
            timestamp: new Date(),
            formData: formData,
            results: results
        };
        
        this.predictionHistory.unshift(historyItem);
        
        // Keep only last 50 predictions
        if (this.predictionHistory.length > 50) {
            this.predictionHistory = this.predictionHistory.slice(0, 50);
        }
        
        // Save to localStorage
        localStorage.setItem('ceramicArmorHistory', JSON.stringify(this.predictionHistory));
    }

    /**
     * Load results history
     */
    loadResultsHistory() {
        // Load from localStorage
        const saved = localStorage.getItem('ceramicArmorHistory');
        if (saved) {
            this.predictionHistory = JSON.parse(saved);
        }

        const historyContainer = document.getElementById('history-container');
        
        if (this.predictionHistory.length === 0) {
            historyContainer.innerHTML = `
                <div class="no-results">
                    <i class="fas fa-info-circle"></i>
                    <p>No prediction history available</p>
                </div>
            `;
            return;
        }

        let historyHTML = '<div class="history-list">';
        
        this.predictionHistory.forEach(item => {
            historyHTML += `
                <div class="history-item" data-id="${item.id}">
                    <div class="history-header">
                        <h4>Prediction ${new Date(item.timestamp).toLocaleString()}</h4>
                        <button class="btn btn-secondary btn-sm" onclick="app.loadHistoryItem(${item.id})">
                            <i class="fas fa-eye"></i> View
                        </button>
                    </div>
                    <div class="history-summary">
                        <p>Composition: ${Object.entries(item.formData.composition)
                            .filter(([k, v]) => v > 0)
                            .map(([k, v]) => `${k}: ${(v*100).toFixed(0)}%`)
                            .join(', ')}</p>
                        <p>Temperature: ${item.formData.processing.sintering_temperature}°C, 
                           Pressure: ${item.formData.processing.pressure} MPa</p>
                    </div>
                </div>
            `;
        });
        
        historyHTML += '</div>';
        historyContainer.innerHTML = historyHTML;
    }

    /**
     * Load a specific history item
     */
    loadHistoryItem(id) {
        const item = this.predictionHistory.find(h => h.id === id);
        if (!item) return;

        // Switch to prediction tab
        this.switchTab('prediction');

        // Load form data
        this.loadFormData(item.formData);

        // Display results
        this.displayResults(item.results, item.formData);

        this.showToast('Historical prediction loaded', 'success');
    }

    /**
     * Load form data into the form
     */
    loadFormData(formData) {
        // Composition
        document.getElementById('sic').value = formData.composition.SiC;
        document.getElementById('b4c').value = formData.composition.B4C;
        document.getElementById('al2o3').value = formData.composition.Al2O3;
        document.getElementById('wc').value = formData.composition.WC || 0;
        document.getElementById('tic').value = formData.composition.TiC || 0;

        // Processing
        document.getElementById('temperature').value = formData.processing.sintering_temperature;
        document.getElementById('pressure').value = formData.processing.pressure;
        document.getElementById('grain-size').value = formData.processing.grain_size;
        document.getElementById('holding-time').value = formData.processing.holding_time || 120;
        document.getElementById('heating-rate').value = formData.processing.heating_rate || 15;
        document.getElementById('atmosphere').value = formData.processing.atmosphere;

        // Microstructure
        document.getElementById('porosity').value = formData.microstructure.porosity;
        document.getElementById('phase-distribution').value = formData.microstructure.phase_distribution;
        document.getElementById('interface-quality').value = formData.microstructure.interface_quality;
        document.getElementById('pore-size').value = formData.microstructure.pore_size || 1.0;

        // Options
        document.getElementById('include-uncertainty').checked = formData.include_uncertainty;
        document.getElementById('include-feature-importance').checked = formData.include_feature_importance;
        document.querySelector(`input[name="prediction-type"][value="${formData.prediction_type}"]`).checked = true;

        // Update sliders and totals
        this.setupSliders();
        this.updateCompositionTotal();
    }

    /**
     * Reset form to default values
     */
    resetForm() {
        // Composition
        document.getElementById('sic').value = 0.6;
        document.getElementById('b4c').value = 0.3;
        document.getElementById('al2o3').value = 0.1;
        document.getElementById('wc').value = 0;
        document.getElementById('tic').value = 0;

        // Processing
        document.getElementById('temperature').value = 1800;
        document.getElementById('pressure').value = 50;
        document.getElementById('grain-size').value = 10;
        document.getElementById('holding-time').value = 120;
        document.getElementById('heating-rate').value = 15;
        document.getElementById('atmosphere').value = 'argon';

        // Microstructure
        document.getElementById('porosity').value = 0.02;
        document.getElementById('phase-distribution').value = 'uniform';
        document.getElementById('interface-quality').value = 'good';
        document.getElementById('pore-size').value = 1.0;

        // Options
        document.getElementById('include-uncertainty').checked = true;
        document.getElementById('include-feature-importance').checked = true;
        document.getElementById('pred-both').checked = true;

        // Update displays
        this.setupSliders();
        this.updateCompositionTotal();

        // Clear results
        document.getElementById('results-container').innerHTML = `
            <div class="no-results">
                <i class="fas fa-info-circle"></i>
                <p>Enter material parameters and click "Predict Properties" to see results</p>
            </div>
        `;

        this.showToast('Form reset to default values', 'success');
    }

    /**
     * Add download buttons for results
     */
    addDownloadButtons(results, formData, container) {
        const downloadContainer = document.createElement('div');
        downloadContainer.className = 'download-section';
        downloadContainer.innerHTML = `
            <h4><i class="fas fa-download"></i> Download Results</h4>
            <div class="download-buttons">
                <button class="btn btn-secondary" onclick="app.downloadResultsCSV()">
                    <i class="fas fa-file-csv"></i> Download CSV
                </button>
                <button class="btn btn-secondary" onclick="app.downloadResultsJSON()">
                    <i class="fas fa-file-code"></i> Download JSON
                </button>
                <button class="btn btn-secondary" onclick="app.downloadResultsPDF()">
                    <i class="fas fa-file-pdf"></i> Download Report (PDF)
                </button>
                <button class="btn btn-secondary" onclick="app.downloadChartsImage()">
                    <i class="fas fa-image"></i> Download Charts
                </button>
            </div>
        `;
        container.appendChild(downloadContainer);

        // Store current results for download
        this.currentResults = { results, formData };
    }

    /**
     * Download results as CSV
     */
    downloadResultsCSV() {
        if (!this.currentResults) return;

        const { results, formData } = this.currentResults;
        let csvContent = "Property,Value,Unit,Lower_CI,Upper_CI,Uncertainty\n";

        // Add mechanical properties
        if (results.mechanical) {
            Object.entries(results.mechanical.predictions).forEach(([key, prop]) => {
                csvContent += `${key},${prop.value},${prop.unit},${prop.confidence_interval[0]},${prop.confidence_interval[1]},${prop.uncertainty}\n`;
            });
        }

        // Add ballistic properties
        if (results.ballistic) {
            Object.entries(results.ballistic.predictions).forEach(([key, prop]) => {
                csvContent += `${key},${prop.value},${prop.unit},${prop.confidence_interval[0]},${prop.confidence_interval[1]},${prop.uncertainty}\n`;
            });
        }

        // Add input parameters
        csvContent += "\n\nInput Parameters\n";
        csvContent += "Parameter,Value\n";
        
        // Composition
        Object.entries(formData.composition).forEach(([key, value]) => {
            if (value > 0) {
                csvContent += `${key}_composition,${value}\n`;
            }
        });

        // Processing
        Object.entries(formData.processing).forEach(([key, value]) => {
            csvContent += `${key},${value}\n`;
        });

        // Microstructure
        Object.entries(formData.microstructure).forEach(([key, value]) => {
            csvContent += `${key},${value}\n`;
        });

        this.downloadFile(csvContent, 'ceramic_armor_predictions.csv', 'text/csv');
    }

    /**
     * Download results as JSON
     */
    downloadResultsJSON() {
        if (!this.currentResults) return;

        const { results, formData } = this.currentResults;
        const jsonData = {
            timestamp: new Date().toISOString(),
            input_parameters: formData,
            predictions: results,
            metadata: {
                prediction_type: formData.prediction_type,
                include_uncertainty: formData.include_uncertainty,
                include_feature_importance: formData.include_feature_importance
            }
        };

        const jsonContent = JSON.stringify(jsonData, null, 2);
        this.downloadFile(jsonContent, 'ceramic_armor_predictions.json', 'application/json');
    }

    /**
     * Download results as PDF report
     */
    downloadResultsPDF() {
        if (!this.currentResults) return;

        // Create a comprehensive HTML report
        const { results, formData } = this.currentResults;
        const reportHTML = this.generateHTMLReport(results, formData);

        // Open in new window for printing/PDF save
        const printWindow = window.open('', '_blank');
        printWindow.document.write(reportHTML);
        printWindow.document.close();
        printWindow.focus();
        
        // Trigger print dialog
        setTimeout(() => {
            printWindow.print();
        }, 500);
    }

    /**
     * Download charts as images
     */
    downloadChartsImage() {
        if (!this.charts || Object.keys(this.charts).length === 0) return;

        // Create a canvas to combine all charts
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Set canvas size
        canvas.width = 1200;
        canvas.height = 800 * Object.keys(this.charts).length;
        
        // White background
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        let yOffset = 0;
        const chartHeight = 800;

        // Draw each chart
        Object.entries(this.charts).forEach(([name, chart]) => {
            if (chart && chart.canvas) {
                ctx.drawImage(chart.canvas, 0, yOffset, 1200, chartHeight);
                yOffset += chartHeight;
            }
        });

        // Download the combined image
        canvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ceramic_armor_charts.png';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
    }

    /**
     * Generate HTML report for PDF export
     */
    generateHTMLReport(results, formData) {
        const timestamp = new Date().toLocaleString();
        
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ceramic Armor ML Prediction Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { text-align: center; margin-bottom: 30px; }
                .section { margin-bottom: 25px; }
                .property-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
                .property-table th, .property-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .property-table th { background-color: #f2f2f2; }
                .input-table { width: 100%; border-collapse: collapse; }
                .input-table th, .input-table td { border: 1px solid #ddd; padding: 6px; }
                .input-table th { background-color: #e8f4f8; }
                .confidence { font-size: 0.9em; color: #666; }
                .uncertainty { font-weight: bold; }
                .uncertainty.excellent { color: #10b981; }
                .uncertainty.good { color: #2563eb; }
                .uncertainty.fair { color: #f59e0b; }
                .uncertainty.poor { color: #ef4444; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Ceramic Armor ML Prediction Report</h1>
                <p>Generated on: ${timestamp}</p>
            </div>

            <div class="section">
                <h2>Input Parameters</h2>
                <h3>Material Composition</h3>
                <table class="input-table">
                    <tr><th>Component</th><th>Fraction</th></tr>
                    ${Object.entries(formData.composition).map(([key, value]) => 
                        value > 0 ? `<tr><td>${key}</td><td>${(value * 100).toFixed(1)}%</td></tr>` : ''
                    ).join('')}
                </table>

                <h3>Processing Parameters</h3>
                <table class="input-table">
                    ${Object.entries(formData.processing).map(([key, value]) => 
                        `<tr><td>${key.replace(/_/g, ' ')}</td><td>${value}</td></tr>`
                    ).join('')}
                </table>

                <h3>Microstructure Parameters</h3>
                <table class="input-table">
                    ${Object.entries(formData.microstructure).map(([key, value]) => 
                        `<tr><td>${key.replace(/_/g, ' ')}</td><td>${value}</td></tr>`
                    ).join('')}
                </table>
            </div>

            ${results.mechanical ? `
            <div class="section">
                <h2>Mechanical Properties Predictions</h2>
                <table class="property-table">
                    <tr>
                        <th>Property</th>
                        <th>Predicted Value</th>
                        <th>Unit</th>
                        <th>95% Confidence Interval</th>
                        <th>Uncertainty</th>
                    </tr>
                    ${Object.entries(results.mechanical.predictions).map(([key, prop]) => {
                        const uncertaintyClass = prop.uncertainty < 0.05 ? 'excellent' : 
                                               prop.uncertainty < 0.1 ? 'good' : 
                                               prop.uncertainty < 0.2 ? 'fair' : 'poor';
                        return `
                        <tr>
                            <td>${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</td>
                            <td>${prop.value.toFixed(3)}</td>
                            <td>${prop.unit}</td>
                            <td class="confidence">[${prop.confidence_interval[0].toFixed(3)}, ${prop.confidence_interval[1].toFixed(3)}]</td>
                            <td class="uncertainty ${uncertaintyClass}">${(prop.uncertainty * 100).toFixed(1)}%</td>
                        </tr>
                        `;
                    }).join('')}
                </table>
            </div>
            ` : ''}

            ${results.ballistic ? `
            <div class="section">
                <h2>Ballistic Properties Predictions</h2>
                <table class="property-table">
                    <tr>
                        <th>Property</th>
                        <th>Predicted Value</th>
                        <th>Unit</th>
                        <th>95% Confidence Interval</th>
                        <th>Uncertainty</th>
                    </tr>
                    ${Object.entries(results.ballistic.predictions).map(([key, prop]) => {
                        const uncertaintyClass = prop.uncertainty < 0.05 ? 'excellent' : 
                                               prop.uncertainty < 0.1 ? 'good' : 
                                               prop.uncertainty < 0.2 ? 'fair' : 'poor';
                        return `
                        <tr>
                            <td>${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</td>
                            <td>${prop.value.toFixed(3)}</td>
                            <td>${prop.unit}</td>
                            <td class="confidence">[${prop.confidence_interval[0].toFixed(3)}, ${prop.confidence_interval[1].toFixed(3)}]</td>
                            <td class="uncertainty ${uncertaintyClass}">${(prop.uncertainty * 100).toFixed(1)}%</td>
                        </tr>
                        `;
                    }).join('')}
                </table>
            </div>
            ` : ''}

            <div class="section">
                <h2>Model Information</h2>
                <p><strong>Prediction Type:</strong> ${formData.prediction_type}</p>
                <p><strong>Uncertainty Quantification:</strong> ${formData.include_uncertainty ? 'Enabled' : 'Disabled'}</p>
                <p><strong>Feature Importance:</strong> ${formData.include_feature_importance ? 'Enabled' : 'Disabled'}</p>
            </div>

            <div class="section">
                <h2>Disclaimer</h2>
                <p><em>These predictions are generated by machine learning models trained on experimental data. 
                Results should be validated through experimental testing before use in critical applications. 
                The confidence intervals represent model uncertainty and may not capture all sources of variability.</em></p>
            </div>
        </body>
        </html>
        `;
    }

    /**
     * Utility function to download files
     */
    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * Load material presets
     */
    loadPresets() {
        this.presets = {
            'sic-dominant': {
                name: 'SiC Dominant Armor',
                composition: { SiC: 0.85, B4C: 0.10, Al2O3: 0.05, WC: 0, TiC: 0 },
                processing: { sintering_temperature: 2000, pressure: 80, grain_size: 5 },
                microstructure: { porosity: 0.015, phase_distribution: 'uniform', interface_quality: 'excellent' }
            },
            'b4c-dominant': {
                name: 'B₄C Dominant Armor',
                composition: { SiC: 0.15, B4C: 0.75, Al2O3: 0.10, WC: 0, TiC: 0 },
                processing: { sintering_temperature: 1900, pressure: 60, grain_size: 8 },
                microstructure: { porosity: 0.02, phase_distribution: 'uniform', interface_quality: 'good' }
            },
            'composite': {
                name: 'Balanced Composite',
                composition: { SiC: 0.4, B4C: 0.3, Al2O3: 0.2, WC: 0.05, TiC: 0.05 },
                processing: { sintering_temperature: 1850, pressure: 70, grain_size: 12 },
                microstructure: { porosity: 0.025, phase_distribution: 'gradient', interface_quality: 'good' }
            }
        };
    }

    /**
     * Show preset selection modal
     */
    showPresetModal() {
        const presetOptions = Object.entries(this.presets).map(([key, preset]) => 
            `<option value="${key}">${preset.name}</option>`
        ).join('');

        const modalHTML = `
            <div class="modal-overlay" id="preset-modal">
                <div class="modal-content">
                    <h3>Load Material Preset</h3>
                    <select id="preset-select">
                        <option value="">Select a preset...</option>
                        ${presetOptions}
                    </select>
                    <div class="modal-buttons">
                        <button class="btn btn-primary" onclick="app.loadSelectedPreset()">Load</button>
                        <button class="btn btn-secondary" onclick="app.closePresetModal()">Cancel</button>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalHTML);
    }

    /**
     * Load selected preset
     */
    loadSelectedPreset() {
        const selectedPreset = document.getElementById('preset-select').value;
        if (!selectedPreset) return;

        const preset = this.presets[selectedPreset];
        
        // Load composition
        Object.entries(preset.composition).forEach(([element, value]) => {
            const elementId = element.toLowerCase().replace('2', '').replace('4', '4');
            const input = document.getElementById(elementId);
            if (input) input.value = value;
        });

        // Load processing (partial)
        document.getElementById('temperature').value = preset.processing.sintering_temperature;
        document.getElementById('pressure').value = preset.processing.pressure;
        document.getElementById('grain-size').value = preset.processing.grain_size;

        // Load microstructure (partial)
        document.getElementById('porosity').value = preset.microstructure.porosity;
        document.getElementById('phase-distribution').value = preset.microstructure.phase_distribution;
        document.getElementById('interface-quality').value = preset.microstructure.interface_quality;

        // Update displays
        this.setupSliders();
        this.updateCompositionTotal();

        this.closePresetModal();
        this.showToast(`Loaded preset: ${preset.name}`, 'success');
    }

    /**
     * Close preset modal
     */
    closePresetModal() {
        const modal = document.getElementById('preset-modal');
        if (modal) modal.remove();
    }

    /**
     * Setup file upload functionality
     */
    setupFileUpload() {
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');

        // Drag and drop handlers
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });

        // File input handler
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files[0]);
            }
        });

        // Click to upload
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
    }

    /**
     * Handle file upload
     */
    handleFileUpload(file) {
        if (!file.name.match(/\.(csv|xlsx)$/i)) {
            this.showToast('Please upload a CSV or Excel file', 'error');
            return;
        }

        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            this.showToast('File size must be less than 10MB', 'error');
            return;
        }

        // Update UI
        document.getElementById('upload-area').innerHTML = `
            <i class="fas fa-file-check"></i>
            <p>File selected: ${file.name}</p>
            <p>Size: ${(file.size / 1024 / 1024).toFixed(2)} MB</p>
        `;

        document.getElementById('process-batch-btn').disabled = false;
        this.uploadedFile = file;

        this.showToast('File uploaded successfully', 'success');
    }

    /**
     * Clear uploaded file
     */
    clearUploadedFile() {
        this.uploadedFile = null;
        document.getElementById('process-batch-btn').disabled = true;
        
        const uploadArea = document.getElementById('upload-area');
        uploadArea.innerHTML = `
            <i class="fas fa-cloud-upload-alt"></i>
            <p>Drag and drop your CSV file here or click to browse</p>
            <input type="file" id="file-input" accept=".csv,.xlsx" hidden>
            <button class="btn btn-secondary" onclick="document.getElementById('file-input').click()">
                Browse Files
            </button>
        `;

        // Re-setup file input
        this.setupFileUpload();
        
        // Clear batch results
        document.getElementById('batch-results').style.display = 'none';
        
        this.showToast('File cleared', 'success');
    }

    /**
     * Process batch predictions
     */
    async processBatch() {
        if (!this.uploadedFile) {
            this.showToast('Please upload a file first', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', this.uploadedFile);
        formData.append('prediction_type', document.getElementById('batch-prediction-type').value);
        formData.append('output_format', document.getElementById('output-format').value);
        formData.append('include_uncertainty', 'true');
        formData.append('include_feature_importance', 'false');

        // Show progress
        document.getElementById('batch-progress').style.display = 'block';
        document.getElementById('process-batch-btn').disabled = true;

        try {
            const response = await axios.post('/api/v1/predict/batch', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                },
                timeout: 300000, // 5 minute timeout for batch processing
                onUploadProgress: (progressEvent) => {
                    const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    this.updateBatchProgress(percentCompleted, 'Uploading...');
                }
            });

            // Handle successful response
            this.handleBatchResults(response.data);
            this.showToast('Batch processing completed successfully!', 'success');

        } catch (error) {
            console.error('Batch processing error:', error);
            this.showToast(`Batch processing failed: ${error.message}`, 'error');
        } finally {
            document.getElementById('batch-progress').style.display = 'none';
            document.getElementById('process-batch-btn').disabled = false;
        }
    }

    /**
     * Update batch processing progress
     */
    updateBatchProgress(percentage, message) {
        const progressFill = document.querySelector('.progress-fill');
        const progressText = document.querySelector('.progress-text');
        
        progressFill.style.width = `${percentage}%`;
        progressText.textContent = `${message} ${percentage}%`;
    }

    /**
     * Handle batch processing results
     */
    handleBatchResults(data) {
        const resultsContainer = document.getElementById('batch-results');
        resultsContainer.style.display = 'block';

        const summary = data.summary || {};
        const processedCount = summary.processed_count || data.processed_count || 0;
        const successCount = summary.success_count || data.successful_count || 0;
        const errorCount = summary.error_count || data.failed_count || 0;

        resultsContainer.innerHTML = `
            <div class="batch-results-content">
                <h3><i class="fas fa-chart-pie"></i> Batch Processing Summary</h3>
                
                <div class="batch-stats">
                    <div class="stat-card">
                        <div class="stat-value">${processedCount}</div>
                        <div class="stat-label">Total Processed</div>
                    </div>
                    <div class="stat-card success">
                        <div class="stat-value">${successCount}</div>
                        <div class="stat-label">Successful</div>
                    </div>
                    <div class="stat-card error">
                        <div class="stat-value">${errorCount}</div>
                        <div class="stat-label">Errors</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${processedCount > 0 ? ((successCount / processedCount) * 100).toFixed(1) : 0}%</div>
                        <div class="stat-label">Success Rate</div>
                    </div>
                </div>

                ${data.preview ? `
                <div class="results-preview">
                    <h4>Results Preview (First 5 rows)</h4>
                    <div class="preview-table-container">
                        ${this.createBatchPreviewTable(data.preview)}
                    </div>
                </div>
                ` : ''}

                <div class="batch-download">
                    <h4><i class="fas fa-download"></i> Download Results</h4>
                    <div class="download-buttons">
                        ${data.download_url ? `
                        <a href="${data.download_url}" class="btn btn-primary" download>
                            <i class="fas fa-file-download"></i> Download Full Results
                        </a>
                        ` : ''}
                        <button class="btn btn-secondary" onclick="app.downloadBatchSummary()">
                            <i class="fas fa-file-alt"></i> Download Summary Report
                        </button>
                        <button class="btn btn-secondary" onclick="app.createBatchVisualization()">
                            <i class="fas fa-chart-bar"></i> View Visualizations
                        </button>
                    </div>
                </div>

                ${errorCount > 0 ? `
                <div class="error-summary">
                    <h4><i class="fas fa-exclamation-triangle"></i> Processing Errors</h4>
                    <p>${errorCount} rows could not be processed. Common issues include:</p>
                    <ul>
                        <li>Invalid composition values (must sum to ≤ 100%)</li>
                        <li>Missing required parameters</li>
                        <li>Values outside acceptable ranges</li>
                    </ul>
                    <p>Check the downloaded results file for detailed error messages.</p>
                </div>
                ` : ''}
            </div>
        `;

        // Store batch results for download and visualization
        this.currentBatchResults = data;
    }

    /**
     * Create preview table for batch results
     */
    createBatchPreviewTable(preview) {
        if (!preview || preview.length === 0) return '<p>No preview data available</p>';

        const headers = Object.keys(preview[0]);
        
        let tableHTML = '<table class="preview-table"><thead><tr>';
        headers.forEach(header => {
            tableHTML += `<th>${header.replace(/_/g, ' ')}</th>`;
        });
        tableHTML += '</tr></thead><tbody>';

        preview.slice(0, 5).forEach(row => {
            tableHTML += '<tr>';
            headers.forEach(header => {
                let value = row[header];
                if (typeof value === 'number') {
                    value = value.toFixed(3);
                }
                tableHTML += `<td>${value}</td>`;
            });
            tableHTML += '</tr>';
        });

        tableHTML += '</tbody></table>';
        return tableHTML;
    }

    /**
     * Download batch summary report
     */
    downloadBatchSummary() {
        if (!this.currentBatchResults) return;

        const data = this.currentBatchResults;
        const summary = data.summary || {};
        
        const reportContent = `
Ceramic Armor ML - Batch Processing Summary Report
Generated: ${new Date().toLocaleString()}

PROCESSING STATISTICS
====================
Total Rows Processed: ${summary.processed_count || data.processed_count || 0}
Successful Predictions: ${summary.success_count || data.successful_count || 0}
Processing Errors: ${summary.error_count || data.failed_count || 0}
Success Rate: ${((summary.success_count || data.successful_count || 0) / (summary.processed_count || data.processed_count || 1) * 100).toFixed(1)}%

PREDICTION STATISTICS
====================
${summary.prediction_stats ? Object.entries(summary.prediction_stats).map(([key, value]) => 
    `${key.replace(/_/g, ' ')}: ${typeof value === 'number' ? value.toFixed(3) : value}`
).join('\n') : 'No prediction statistics available'}

PROCESSING DETAILS
==================
Prediction Type: ${data.prediction_type || 'Unknown'}
Output Format: ${data.output_format || 'Unknown'}
Processing Time: ${summary.processing_time || 'Unknown'}

${(summary.error_count || data.failed_count || 0) > 0 ? `
COMMON ERRORS
=============
- Invalid composition values (must sum to ≤ 100%)
- Missing required parameters
- Values outside acceptable ranges
- Invalid file format or structure

Please check the full results file for detailed error messages.
` : ''}

DISCLAIMER
==========
These predictions are generated by machine learning models trained on experimental data.
Results should be validated through experimental testing before use in critical applications.
        `;

        this.downloadFile(reportContent, 'batch_processing_summary.txt', 'text/plain');
    }

    /**
     * Create batch visualization
     */
    createBatchVisualization() {
        if (!this.currentBatchResults) return;

        // Switch to results tab and create visualizations
        this.switchTab('results');
        
        // Create batch visualization charts
        this.createBatchCharts(this.currentBatchResults);
    }

    /**
     * Create charts for batch results
     */
    createBatchCharts(data) {
        const historyContainer = document.getElementById('history-container');
        historyContainer.innerHTML = `
            <div class="batch-visualization">
                <h3><i class="fas fa-chart-line"></i> Batch Results Visualization</h3>
                <div id="batch-charts-container" class="charts-container"></div>
            </div>
        `;

        const chartsContainer = document.getElementById('batch-charts-container');

        // Create distribution charts if we have prediction statistics
        if (data.summary?.prediction_stats) {
            this.createBatchDistributionChart(data.summary.prediction_stats, chartsContainer);
        }

        // Create success rate chart
        this.createBatchSuccessChart(data.summary || data, chartsContainer);
    }

    /**
     * Create distribution chart for batch predictions
     */
    createBatchDistributionChart(stats, container) {
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        chartContainer.innerHTML = `
            <h4><i class="fas fa-chart-histogram"></i> Prediction Value Distributions</h4>
            <canvas id="batch-distribution-chart"></canvas>
        `;
        container.appendChild(chartContainer);

        const ctx = document.getElementById('batch-distribution-chart').getContext('2d');

        // Extract property statistics
        const properties = Object.keys(stats).filter(key => key.includes('_mean') || key.includes('_avg'));
        const labels = properties.map(prop => prop.replace(/_mean|_avg/g, '').replace(/_/g, ' '));
        const means = properties.map(prop => stats[prop]);
        const stds = properties.map(prop => stats[prop.replace('_mean', '_std').replace('_avg', '_std')] || 0);

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Mean Value',
                        data: means,
                        backgroundColor: 'rgba(37, 99, 235, 0.6)',
                        borderColor: 'rgba(37, 99, 235, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Standard Deviation',
                        data: stds,
                        backgroundColor: 'rgba(239, 68, 68, 0.6)',
                        borderColor: 'rgba(239, 68, 68, 1)',
                        borderWidth: 1,
                        type: 'line',
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Statistical Distribution of Predicted Properties'
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Mean Value'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Standard Deviation'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    },
                    x: {
                        ticks: {
                            maxRotation: 45
                        }
                    }
                }
            }
        });
    }

    /**
     * Create success rate chart for batch processing
     */
    createBatchSuccessChart(summary, container) {
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        chartContainer.innerHTML = `
            <h4><i class="fas fa-chart-pie"></i> Processing Success Rate</h4>
            <canvas id="batch-success-chart"></canvas>
        `;
        container.appendChild(chartContainer);

        const ctx = document.getElementById('batch-success-chart').getContext('2d');

        const successCount = summary.success_count || summary.successful_count || 0;
        const errorCount = summary.error_count || summary.failed_count || 0;

        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Successful', 'Errors'],
                datasets: [{
                    data: [successCount, errorCount],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.8)',
                        'rgba(239, 68, 68, 0.8)'
                    ],
                    borderColor: [
                        'rgba(16, 185, 129, 1)',
                        'rgba(239, 68, 68, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Processing Results (${successCount + errorCount} total)`
                    },
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    /**
     * Show loading overlay
     */
    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        overlay.style.display = show ? 'flex' : 'none';
    }

    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <i class="fas fa-${this.getToastIcon(type)}"></i>
            <span>${message}</span>
        `;

        document.getElementById('toast-container').appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            toast.remove();
        }, 5000);
    }

    /**
     * Get toast icon based on type
     */
    getToastIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new CeramicArmorApp();
});

// Add modal styles dynamically
const modalStyles = `
<style>
.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}

.modal-content {
    background: var(--surface-color);
    padding: 2rem;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-lg);
    min-width: 400px;
    max-width: 90vw;
}

.modal-content h3 {
    margin-bottom: 1rem;
    color: var(--text-primary);
}

.modal-content select {
    width: 100%;
    margin-bottom: 1.5rem;
}

.modal-buttons {
    display: flex;
    gap: 1rem;
    justify-content: flex-end;
}

.history-item {
    background: var(--surface-color);
    padding: 1.5rem;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-sm);
    margin-bottom: 1rem;
}

.history-header {
    display: flex;
    justify-content: between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.history-header h4 {
    margin: 0;
    color: var(--text-primary);
}

.history-summary {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.batch-summary {
    background: var(--background-color);
    padding: 1rem;
    border-radius: var(--border-radius);
    margin-bottom: 1rem;
}

.download-section {
    text-align: center;
}

.btn-sm {
    padding: 0.5rem 1rem;
    font-size: 0.8rem;
}
</style>
`;

document.head.insertAdjacentHTML('beforeend', modalStyles);