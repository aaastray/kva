import { defineStore } from 'pinia';
import { type Patient, type Analysis } from '@/types';

export const usePatientsStore = defineStore('patients', {
    state: () => ({
        patients: [] as Patient[],
        nextPatientId: 1,
        nextAnalysisId: 1
    }),

    getters: {
        getPatientById: (state) => (id: number) => {
            return state.patients.find(patient => patient.id === id);
        },

        getAnalysisByIds: (state) => (patientId: number, analysisId: number) => {
            const patient = state.patients.find(p => p.id === patientId);
            if (!patient) return null;
            return patient.analyses.find(a => a.id === analysisId);
        }
    },

    actions: {
        addPatient(patient: Omit<Patient, 'id' | 'analyses'>) {
            const newPatient: Patient = {
                ...patient,
                id: this.nextPatientId++,
                analyses: []
            };
            this.patients.push(newPatient);
            return newPatient.id;
        },

        addAnalysis(patientId: number, analysisData: Omit<Analysis, 'id' | 'patientId'>) {
            const patient = this.patients.find(p => p.id === patientId);
            if (!patient) return null;

            const newAnalysis: Analysis = {
                ...analysisData,
                id: this.nextAnalysisId++,
                patientId
            };

            patient.analyses.push(newAnalysis);
            return newAnalysis.id;
        },

        updateAnalysisNotes(patientId: number, analysisId: number, notes: string) {
            const patient = this.patients.find(p => p.id === patientId);
            if (!patient) return false;

            const analysis = patient.analyses.find(a => a.id === analysisId);
            if (!analysis) return false;

            analysis.doctorNotes = notes;
            return true;
        },

        // Метод для инициализации тестовых данных
        initDemoData() {
            // Добавляем тестовых пациентов
            const patient1Id = this.addPatient({
                lastName: 'Иванов',
                firstName: 'Иван',
                middleName: 'Иванович',
                birthDate: '2010-05-15',
                gender: 'male',
                policyNumber: '1234567890'
            });

            const patient2Id = this.addPatient({
                lastName: 'Петрова',
                firstName: 'Мария',
                middleName: 'Сергеевна',
                birthDate: '2012-08-23',
                gender: 'female',
                policyNumber: '0987654321'
            });

            // Добавляем тестовые анализы
            this.addAnalysis(patient1Id, {
                date: '2023-06-10',
                predictedAge: 13.2,
                xrayImage: 'https://via.placeholder.com/500x500?text=X-ray+Image+1',
                doctorNotes: 'Развитие соответствует возрасту'
            });

            this.addAnalysis(patient1Id, {
                date: '2024-01-15',
                predictedAge: 13.8,
                xrayImage: 'https://via.placeholder.com/500x500?text=X-ray+Image+2',
                doctorNotes: 'Наблюдается нормальный рост'
            });

            this.addAnalysis(patient2Id, {
                date: '2023-09-05',
                predictedAge: 11.5,
                xrayImage: 'https://via.placeholder.com/500x500?text=X-ray+Image+3',
                doctorNotes: 'Костный возраст немного отстает от паспортного'
            });
        }
    }
});