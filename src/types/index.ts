export interface Patient {
    id: number;
    lastName: string;
    firstName: string;
    middleName: string;
    birthDate: string;
    gender: 'male' | 'female';
    policyNumber: string;
    analyses: Analysis[];
}

export interface Analysis {
    id: number;
    patientId: number;
    date: string;
    predictedAge: number;
    xrayImage: string;
    doctorNotes?: string;
}

export interface PatientFilter {
    searchQuery: string;
}