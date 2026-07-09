from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, DecimalField, SelectField
from wtforms.validators import DataRequired, Email, Length, EqualTo, NumberRange, Optional

# -------------------------------------------------------------
# 1. AUTHENTICATION FORMS
# -------------------------------------------------------------
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=80)])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=80)])
    email = StringField('Email Address', validators=[DataRequired(), Email(), Length(max=120)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6, max=128)])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(), 
        EqualTo('password', message='Passwords must match.')
    ])
    submit = SubmitField('Register')

class ForgotPasswordForm(FlaskForm):
    email = StringField('Email Address', validators=[DataRequired(), Email()])
    submit = SubmitField('Submit')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('New Password', validators=[DataRequired(), Length(min=6, max=128)])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message='Passwords must match.')
    ])
    submit = SubmitField('Update Password')

# -------------------------------------------------------------
# 2. USER PROFILE FORM
# -------------------------------------------------------------
class ProfileForm(FlaskForm):
    theme_preference = SelectField('Theme Preference', choices=[
        ('light', 'Light Mode'),
        ('dark', 'Dark Mode')
    ], validators=[DataRequired()])
    old_password = PasswordField('Current Password', validators=[Optional()])
    new_password = PasswordField('New Password', validators=[Optional(), Length(min=6, max=128)])
    submit = SubmitField('Save Profile')

# -------------------------------------------------------------
# 3. METEOROLOGICAL PREDICTION PARAMETERS FORM
# -------------------------------------------------------------
class PredictForm(FlaskForm):
    location = StringField('Location Name', validators=[DataRequired(), Length(max=100)])
    Latitude = DecimalField('Latitude (°N)', places=4, default=20.5937, validators=[
        DataRequired(), 
        NumberRange(min=-90.0, max=90.0)
    ])
    Longitude = DecimalField('Longitude (°E)', places=4, default=78.9629, validators=[
        DataRequired(), 
        NumberRange(min=-180.0, max=180.0)
    ])
    
    # Environmental variables
    Annual_Rainfall = DecimalField('Annual Rainfall (mm)', places=1, validators=[
        DataRequired(), 
        NumberRange(min=100.0, max=10000.0)
    ])
    Monthly_Rainfall = DecimalField('Monthly Rainfall (mm)', places=1, validators=[
        DataRequired(), 
        NumberRange(min=0.0, max=2500.0)
    ])
    Temperature = DecimalField('Temperature (°C)', places=1, validators=[
        DataRequired(), 
        NumberRange(min=-10.0, max=60.0)
    ])
    Humidity = DecimalField('Relative Humidity (%)', places=1, validators=[
        DataRequired(), 
        NumberRange(min=10.0, max=100.0)
    ])
    Pressure = DecimalField('Atmospheric Pressure (hPa)', places=1, validators=[
        DataRequired(), 
        NumberRange(min=900.0, max=1100.0)
    ])
    Cloud_Cover = DecimalField('Cloud Cover (%)', places=1, validators=[
        DataRequired(), 
        NumberRange(min=0.0, max=100.0)
    ])
    Wind_Speed = DecimalField('Wind Speed (km/h)', places=1, validators=[
        DataRequired(), 
        NumberRange(min=0.0, max=150.0)
    ])
    River_Water_Level = DecimalField('River Water Level (m)', places=2, validators=[
        DataRequired(), 
        NumberRange(min=0.0, max=30.0)
    ])
    Ground_Water_Level = DecimalField('Ground Water Level (m)', places=2, validators=[
        DataRequired(), 
        NumberRange(min=0.0, max=30.0)
    ])
    Visibility = DecimalField('Visibility (km)', places=1, validators=[
        DataRequired(), 
        NumberRange(min=0.0, max=50.0)
    ])
    
    # Categoricals
    Season = SelectField('Season', choices=[
        ('', 'Select Season...'),
        ('Summer', 'Summer'),
        ('Monsoon', 'Monsoon'),
        ('Winter', 'Winter'),
        ('Spring', 'Spring')
    ], validators=[DataRequired()])
    
    Month = SelectField('Month', choices=[
        ('', 'Select Month...'),
        ('January', 'January'), ('February', 'February'), ('March', 'March'),
        ('April', 'April'), ('May', 'May'), ('June', 'June'),
        ('July', 'July'), ('August', 'August'), ('September', 'September'),
        ('October', 'October'), ('November', 'November'), ('December', 'December')
    ], validators=[DataRequired()])
    
    District = StringField('District', default='Unknown', validators=[Optional(), Length(max=50)])
    State = StringField('State', default='Unknown', validators=[Optional(), Length(max=50)])
    
    submit = SubmitField('Analyze Risk')
