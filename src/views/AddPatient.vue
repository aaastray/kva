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
      >
        <el-row :gutter="20">
          <el-col :span="8">
            <el-form-item label="Фамилия" prop="lastName">
              <el-input v-model="patientForm.lastName" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="Имя" prop="firstName">
              <el-input v-model="patientForm.firstName" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="Отчество" prop="middleName">
              <el-input v-model="patientForm.middleName" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <el-col :span="8">
            <el-form-item label="Дата рождения" prop="birthDate">
              <el-date-picker
                  v-model="patientForm.birthDate"
                  type="date"
                  placeholder="Выберите дату"
                  style="width: 100%"
                  format="DD.MM.YYYY"
                  value-format="YYYY-MM-DD"
              />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="Пол" prop="gender">
              <el-radio-group v-model="patientForm.gender">
                <el-radio label="male">Мужской</el-radio>
                <el-radio label="female">Женский</el-radio>
              </el-radio-group>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="Номер полиса" prop="policyNumber">
              <el-input v-model="patientForm.policyNumber" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-form-item>
          <el-button type="primary" @click="submitForm(formRef)">
            Сохранить
          </el-button>
          <el-button @click="resetForm(formRef)">Очистить</el-button>
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
import { usePatientsStore } from '@/stores/patients';

const router = useRouter();
const patientsStore = usePatientsStore();
const formRef = ref<FormInstance>();

const patientForm = reactive({
  lastName: '',
  firstName: '',
  middleName: '',
  birthDate: '',
  gender: 'male' as 'male' | 'female',
  policyNumber: ''
});

const rules: FormRules = {
  lastName: [
    { required: true, message: 'Пожалуйста, введите фамилию', trigger: 'blur' },
    { min: 2, message: 'Минимум 2 символа', trigger: 'blur' }
  ],
  firstName: [
    { required: true, message: 'Пожалуйста, введите имя', trigger: 'blur' },
    { min: 2, message: 'Минимум 2 символа', trigger: 'blur' }
  ],
  middleName: [
    { min: 2, message: 'Минимум 2 символа', trigger: 'blur' }
  ],
  birthDate: [
    { required: true, message: 'Пожалуйста, выберите дату рождения', trigger: 'change' }
  ],
  gender: [
    { required: true, message: 'Пожалуйста, выберите пол', trigger: 'change' }
  ],
  policyNumber: [
    { required: true, message: 'Пожалуйста, введите номер полиса', trigger: 'blur' },
    { min: 6, message: 'Минимум 6 символов', trigger: 'blur' }
  ]
};

const submitForm = async (formEl: FormInstance | undefined) => {
  if (!formEl) return;

  await formEl.validate((valid) => {
    if (valid) {
      const patientId = patientsStore.addPatient(patientForm);
      ElMessage({
        message: 'Пациент успешно добавлен',
        type: 'success'
      });
      router.push({ name: 'patient', params: { id: patientId } });
    } else {
      ElMessage({
        message: 'Пожалуйста, заполните все обязательные поля',
        type: 'error'
      });
    }
  });
};

const resetForm = (formEl: FormInstance | undefined) => {
  if (!formEl) return;
  formEl.resetFields();
};

const goBack = () => {
  router.push({ name: 'home' });
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