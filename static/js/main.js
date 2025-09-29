// ===== MAIN APPLICATION ===== 
class PEPerformancePredictor {
    constructor() {
        this.form = document.getElementById('studentForm');
        this.submitBtn = document.getElementById('submitBtn');
        this.resetBtn = document.getElementById('resetBtn');
        this.autoFillBtn = document.getElementById('autoFillBtn');
        this.resultSection = document.getElementById('resultSection');
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.setupFormValidation();
    }

    bindEvents() {
        this.form.addEventListener('submit', this.handleFormSubmit.bind(this));
        this.resetBtn.addEventListener('click', this.handleReset.bind(this));
        this.autoFillBtn.addEventListener('click', this.handleAutoFill.bind(this));
        
        // Add input validation listeners
        const inputs = this.form.querySelectorAll('.input-field');
        inputs.forEach(input => {
            input.addEventListener('blur', this.validateField.bind(this, input));
            input.addEventListener('input', this.clearFieldError.bind(this, input));
        });
    }

    setupFormValidation() {
        // Custom validation rules for specific fields
        this.validationRules = {
            Age: { min: 14, max: 17, type: 'number' },
            Strength_Score: { min: 0, max: 100, type: 'number' },
            Endurance_Score: { min: 0, max: 100, type: 'number' },
            Flexibility_Score: { min: 0, max: 100, type: 'number' },
            Speed_Agility_Score: { min: 0, max: 100, type: 'number' },
            BMI: { min: 10, max: 50, type: 'number' },
            Skills_Score: { min: 0, max: 100, type: 'number' },
            Health_Fitness_Knowledge_Score: { min: 0, max: 100, type: 'number' },
            Attendance_Rate: { min: 0, max: 100, type: 'number' },
            Overall_PE_Performance_Score: { min: 0, max: 100, type: 'number' },
            Improvement_Rate: { min: 0, max: 20, type: 'number' },
            Hours_Physical_Activity_Per_Week: { min: 0, max: 50, type: 'number' }
        };
    }

    validateField(input) {
        const fieldName = input.name;
        const value = input.value.trim();
        const rules = this.validationRules[fieldName];
        
        this.clearFieldError(input);

        if (!value && input.required) {
            this.showFieldError(input, 'Questo campo è obbligatorio');
            return false;
        }

        if (rules && rules.type === 'number' && value) {
            const numValue = parseFloat(value);
            if (isNaN(numValue)) {
                this.showFieldError(input, 'Inserisci un numero valido');
                return false;
            }
            if (numValue < rules.min || numValue > rules.max) {
                this.showFieldError(input, `Il valore deve essere compreso tra ${rules.min} e ${rules.max}`);
                return false;
            }
        }

        return true;
    }

    showFieldError(input, message) {
        input.classList.add('error');
        
        // Remove existing error message
        const existingError = input.parentNode.querySelector('.field-error');
        if (existingError) {
            existingError.remove();
        }
        
        // Add new error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'field-error';
        errorDiv.textContent = message;
        input.parentNode.appendChild(errorDiv);
        
        // Add error styles to CSS if not already present
        this.addErrorStyles();
    }

    clearFieldError(input) {
        input.classList.remove('error');
        const errorDiv = input.parentNode.querySelector('.field-error');
        if (errorDiv) {
            errorDiv.remove();
        }
    }

    addErrorStyles() {
        const styleId = 'error-styles';
        if (document.getElementById(styleId)) return;
        
        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = `
            .input-field.error {
                border-color: var(--color-error);
                box-shadow: 0 0 0 3px var(--color-error-light);
            }
            .field-error {
                color: var(--color-error);
                font-size: var(--font-size-xs);
                margin-top: var(--space-xs);
                font-weight: 500;
            }
        `;
        document.head.appendChild(style);
    }

    async handleFormSubmit(e) {
        e.preventDefault();
        
        // Validate all fields
        const inputs = this.form.querySelectorAll('.input-field');
        let isValid = true;
        
        inputs.forEach(input => {
            if (!this.validateField(input)) {
                isValid = false;
            }
        });

        if (!isValid) {
            this.showNotification('Correggi gli errori nel form prima di continuare', 'error');
            return;
        }

        await this.submitPrediction();
    }

    async submitPrediction() {
        this.setLoadingState(true);
        
        try {
            const formData = new FormData(this.form);
            const data = Object.fromEntries(formData);
            
            // Convert numeric fields
            Object.keys(data).forEach(key => {
                if (this.validationRules[key] && this.validationRules[key].type === 'number') {
                    data[key] = parseFloat(data[key]);
                }
            });
            
            console.log('Sending data:', data); // Debug log
            
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || 'Errore nella previsione');
            }

            console.log('Received result:', result); // Debug log
            this.showResult(result);
            
        } catch (error) {
            console.error('Error:', error);
            this.showNotification(
                error.message || 'Errore di connessione. Riprova più tardi.',
                'error'
            );
        } finally {
            this.setLoadingState(false);
        }
    }

    setLoadingState(isLoading) {
        const btnText = this.submitBtn.querySelector('.btn-text');
        const btnLoader = this.submitBtn.querySelector('.btn-loader');
        
        if (isLoading) {
            this.submitBtn.disabled = true;
            btnText.style.display = 'none';
            btnLoader.style.display = 'block';
        } else {
            this.submitBtn.disabled = false;
            btnText.style.display = 'inline';
            btnLoader.style.display = 'none';
        }
    }

    showResult(result) {
        const { prediction_italian, confidence } = result;
        
        // Update result card
        this.updateResultCard(prediction_italian, confidence);
        
        // Show result section
        this.resultSection.style.display = 'block';
        this.resultSection.scrollIntoView({ behavior: 'smooth' });
        
        this.showNotification('Previsione completata con successo!', 'success');
    }

    updateResultCard(prediction, confidence) {
        const performanceValue = document.getElementById('performanceValue');
        const resultDescription = document.getElementById('resultDescription');
        const confidenceValue = document.getElementById('confidenceValue');
        const confidenceFill = document.getElementById('confidenceFill');
        
        // Determine performance level and description
        let performanceClass = 'medium';
        let displayValue = prediction;
        let description = 'Basato sui dati forniti, il modello prevede una prestazione media.';
        
        if (prediction.includes('Alta') || prediction.includes('High')) {
            performanceClass = 'high';
            displayValue = 'Alta';
            description = 'Basato sui dati forniti, il modello prevede una prestazione alta in educazione fisica.';
        } else if (prediction.includes('Bassa') || prediction.includes('Low')) {
            performanceClass = 'low';
            displayValue = 'Bassa';
            description = 'Basato sui dati forniti, il modello prevede una prestazione bassa in educazione fisica.';
        } else if (prediction.includes('Media') || prediction.includes('Average')) {
            performanceClass = 'medium';
            displayValue = 'Media';
            description = 'Basato sui dati forniti, il modello prevede una prestazione media in educazione fisica.';
        }
        
        // Update content
        performanceValue.textContent = displayValue;
        performanceValue.className = `performance-value ${performanceClass}`;
        resultDescription.textContent = description;
        
        // Update confidence
        const confidencePercent = Math.round(confidence * 100);
        confidenceValue.textContent = `${confidencePercent}%`;
        confidenceFill.style.width = `${confidencePercent}%`;
    }

    handleReset() {
        if (confirm('Sei sicuro di voler cancellare tutti i campi?')) {
            this.form.reset();
            this.clearAllErrors();
            this.showNotification('Campi cancellati', 'info');
        }
    }

    handleAutoFill() {
        // Valori realistici per auto-fill - TUTTI I CAMPI inclusi dropdown
        const randomValues = {
            // Dropdown fields
            Age: ['14', '15', '16', '17'],
            Gender: ['Male', 'Female', 'Other'],
            Grade_Level: ['9th', '10th', '11th', '12th'],
            Class_Participation_Level: ['Low', 'Medium', 'High'],
            Motivation_Level: ['Low', 'Medium', 'High'],
            Final_Grade: ['A', 'B', 'C'],
            Previous_Semester_PE_Grade: ['A', 'B', 'C'],
            
            // Numeric input fields
            Strength_Score: () => Math.floor(Math.random() * 40) + 40, // 40-80
            Endurance_Score: () => Math.floor(Math.random() * 40) + 45, // 45-85
            Flexibility_Score: () => Math.floor(Math.random() * 35) + 50, // 50-85
            Speed_Agility_Score: () => Math.floor(Math.random() * 40) + 35, // 35-75
            BMI: () => (Math.random() * 8 + 18).toFixed(1), // 18.0-26.0
            Skills_Score: () => Math.floor(Math.random() * 35) + 50, // 50-85
            Health_Fitness_Knowledge_Score: () => Math.floor(Math.random() * 40) + 40, // 40-80
            Attendance_Rate: () => Math.floor(Math.random() * 25) + 70, // 70-95%
            Grade_Math: () => (Math.random() * 3 + 6).toFixed(1), // 6.0-9.0
            Grade_Science: () => (Math.random() * 3 + 6).toFixed(1), // 6.0-9.0
            Overall_PE_Performance_Score: () => Math.floor(Math.random() * 30) + 60, // 60-90
            Improvement_Rate: () => (Math.random() * 8 + 2).toFixed(1), // 2.0-10.0
            Hours_Physical_Activity_Per_Week: () => Math.floor(Math.random() * 12) + 3 // 3-15 ore
        };

        // Applica i valori random ai campi
        Object.keys(randomValues).forEach(fieldName => {
            const field = document.querySelector(`[name="${fieldName}"]`);
            if (field) {
                let value;
                if (Array.isArray(randomValues[fieldName])) {
                    // Selezione casuale da array
                    value = randomValues[fieldName][Math.floor(Math.random() * randomValues[fieldName].length)];
                } else if (typeof randomValues[fieldName] === 'function') {
                    // Esecuzione funzione per valore calcolato
                    value = randomValues[fieldName]();
                }
                
                field.value = value;
                this.clearFieldError(field);
            }
        });

        this.showNotification('Campi compilati automaticamente!', 'success');
    }

    handleNewPrediction() {
        this.resultSection.style.display = 'none';
        this.form.scrollIntoView({ behavior: 'smooth' });
    }

    clearAllErrors() {
        const inputs = this.form.querySelectorAll('.input-field');
        inputs.forEach(input => this.clearFieldError(input));
    }

    showNotification(message, type = 'info') {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.notification');
        existingNotifications.forEach(n => n.remove());
        
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Add notification styles
        this.addNotificationStyles();
        
        document.body.appendChild(notification);
        
        // Auto-remove after 4 seconds
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => notification.remove(), 300);
        }, 4000);
    }

    addNotificationStyles() {
        const styleId = 'notification-styles';
        if (document.getElementById(styleId)) return;
        
        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = `
            .notification {
                position: fixed;
                top: var(--space-xl);
                right: var(--space-xl);
                background: var(--color-bg);
                border: 1px solid var(--color-border);
                border-radius: var(--border-radius-md);
                padding: var(--space-md) var(--space-lg);
                box-shadow: var(--shadow-lg);
                z-index: 1000;
                max-width: 300px;
                font-size: var(--font-size-sm);
                animation: slideIn 0.3s ease;
            }
            .notification-success {
                border-color: var(--color-success);
                background-color: var(--color-success-light);
                color: var(--color-success);
            }
            .notification-error {
                border-color: var(--color-error);
                background-color: var(--color-error-light);
                color: var(--color-error);
            }
            .notification-info {
                border-color: var(--color-primary);
                background-color: var(--color-primary-light);
                color: var(--color-primary);
            }
            .notification.fade-out {
                animation: slideOut 0.3s ease;
            }
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
}

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    new PEPerformancePredictor();
});

// ===== UTILS =====
// Add some helper functions for better UX
document.addEventListener('keydown', (e) => {
    // ESC key to close result section
    if (e.key === 'Escape') {
        const resultSection = document.getElementById('resultSection');
        if (resultSection.style.display !== 'none') {
            resultSection.style.display = 'none';
        }
    }
});

// Smooth scroll polyfill for older browsers
if (!('scrollBehavior' in document.documentElement.style)) {
    const smoothScrollPolyfill = document.createElement('script');
    smoothScrollPolyfill.src = 'https://cdn.jsdelivr.net/gh/iamdustan/smoothscroll@1.4.10/src/smoothscroll.js';
    document.head.appendChild(smoothScrollPolyfill);
}