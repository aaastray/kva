<template>
  <div class="add-patient">
    <div class="page-header">
      <h2>Добавление нового пациента</h2>
      <el-button type="primary" @click="goBack">
        <el-icon><Back /></el-icon>
        Назад к списку
      </el-button>
    </div>

    <el-card>
      <el-form
          ref="formRef"
          :model="patientForm"
          :rules="rules"
          label-position="top"
          status-icon
          @submit.prevent="submitForm(formRef)"
      >
        <el-row :gutter="20">
          <el-col :xs="24" :sm="12" :md="8">
            <el-form-item label="Фамилия" prop="lastName">
              <el-input v-model="patientForm.lastName" />
            </el-form-item>
          </el-col>
          <el-col :xs="24" :sm="12" :md="8">
            <el-form-item label="Имя" prop="firstName">
              <el-input v-model="patientForm.firstName" />
            </el-form-item>
          </el-col>
          <el-col :xs="24" :sm="12" :md="8">
            <el-form-item label="Отчество" prop="middleName">
              <el-input v-model="patientForm.middleName" placeholder="Необязательно" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <el-col :xs="24" :sm="12" :md="8">
            <el-form-item label="Дата рождения" prop="birthDate">
              <el-date-picker
                  v-model="patientForm.birthDate"
                  type="date"
                  placeholder="Выберите дату"
                  style="width: 100%"
                  format="DD.MM.YYYY"
                  value-format="YYYY-MM-DD"
                  :disabled-date="disabledFutureDates"
              />
            </el-form-item>
          </el-col>
          <el-col :xs="24" :sm="12" :md="8">
            <el-form-item label="Пол" prop="gender">
              <el-radio-group v-model="patientForm.gender">
                <el-radio label="male">Мужской</el-radio>
                <el-radio label="female">Женский</el-radio>
              </el-radio-group>
            </el-form-item>
          </el-col>
          <el-col :xs="24" :sm="12" :md="8">
            <el-form-item label="Номер полиса" prop="policyNumber">
              <el-input v-model="patientForm.policyNumber" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-form-item>
          <el-button type="primary" @click="submitForm(formRef)" :loading="isSubmitting">
            Сохранить
          </el-button>
          <el-button @click="resetForm(formRef)" :disabled="isSubmitting">Очистить</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { reactive, ref } from 'vue';
import { useRouter } from 'vue-router';
import type { FormInstance, FormRules } from 'element-plus';
import { ElMessage } from 'element-plus';
import { Back } from '@element-plus/icons-vue';
import apiClient from '@/services/api'; // Убедитесь, что путь правильный

interface PatientCreatePayload {
  lastName: string;
  firstName: string;
  middleName?: string;
  birthDate: string; // YYYY-MM-DD
  gender: 'male' | 'female';
  policyNumber: string;
}

interface PatientBackend { // Тип ответа от сервера
  id: number;
  lastName: string;
  firstName: string;
  middleName?: string;
  birthDate: string;
  gender: 'male' | 'female';
  policyNumber: string;
  analyses: any[];
}

const router = useRouter();
const formRef = ref<FormInstance>();
const isSubmitting = ref(false);

const patientForm = reactive<PatientCreatePayload>({
  lastName: '',
  firstName: '',
  middleName: '',
  birthDate: '',
  gender: 'male',
  policyNumber: ''
});

const rules: FormRules = {
  lastName: [
    { required: true, message: 'Пожалуйста, введите фамилию', trigger: 'blur' },
    { min: 2, max: 50, message: 'Длина от 2 до 50 символов', trigger: 'blur' }
  ],
  firstName: [
    { required: true, message: 'Пожалуйста, введите имя', trigger: 'blur' },
    { min: 2, max: 50, message: 'Длина от 2 до 50 символов', trigger: 'blur' }
  ],
  middleName: [
    { min: 2, max: 50, message: 'Длина от 2 до 50 символов', trigger: 'blur' }
  ],
  birthDate: [
    { required: true, message: 'Пожалуйста, выберите дату рождения', trigger: 'change' }
  ],
  gender: [
    { required: true, message: 'Пожалуйста, выберите пол', trigger: 'change' }
  ],
  policyNumber: [
    { required: true, message: 'Пожалуйста, введите номер полиса', trigger: 'blur' },
    { pattern: /^[0-9]+$/, message: 'Номер полиса должен содержать только цифры', trigger: 'blur'},
    { min: 6, max: 20, message: 'Длина от 6 до 20 символов', trigger: 'blur' }
  ]
};

const disabledFutureDates = (time: Date) => {
  return time.getTime() > Date.now();
};

const submitForm = async (formEl: FormInstance | undefined) => {
  if (!formEl) return;
  isSubmitting.value = true;

  await formEl.validate(async (valid) => {
    if (valid) {
      try {
        const payload: PatientCreatePayload = {
          ...patientForm,
          middleName: patientForm.middleName || undefined, // Отправляем undefined если пусто
        };
        const response = await apiClient.post<PatientBackend>('/patients', payload);
        const newPatient = response.data;

        ElMessage({
          message: `Пациент ${newPatient.lastName} ${newPatient.firstName} успешно добавлен!`,
          type: 'success'
        });
        router.push({ name: 'patient', params: { id: newPatient.id } });
      } catch (error: any) {
         ElMessage({
          message: `Ошибка при добавлении пациента: ${error.message || 'Ошибка сервера'}`,
          type: 'error'
        });
      } finally {
        isSubmitting.value = false;
      }
    } else {
      ElMessage({
        message: 'Пожалуйста, проверьте правильность заполнения полей.',
        type: 'error'
      });
      isSubmitting.value = false;
    }
  });
};

const resetForm = (formEl: FormInstance | undefined) => {
  if (!formEl) return;
  formEl.resetFields();
  patientForm.middleName = ''; // resetFields может не сбрасывать не-required поля, если они не были undefined изначально
};

const goBack = () => {
  router.push({ name: 'home' }); // Предполагается, что 'home' - это имя роута списка пациентов
};
</script>

<style scoped>
.add-patient {
  padding: 20px 0;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.el-card {
  margin-bottom: 20px;
}
</style>