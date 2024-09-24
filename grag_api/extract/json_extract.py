import json
import os
from collections import defaultdict
import re


def process_json_element(element):
    description = f"Job Role: {element.get('jobRole', 'N/A')}\n"
    description += f"Sector: {element.get('sector', 'N/A')}\n"
    description += f"Sub-Sector: {element.get('subSector', 'N/A')}\n"
    description += f"College Category: {element.get('collegeCategory', 'N/A')}\n"
    description += f"Job Location: {element.get('jobLocation', 'N/A')}\n"
    description += f"Experience Level: {element.get('experienceLevel', 'N/A')}\n\n"

    description += "Job Profile:\n"
    if 'jobProfile' in element:
        job_profile = element['jobProfile']
        description += f"- General Description: {job_profile.get('generalDescription', {}).get('text', 'N/A')}\n"
        description += f"- Day in the Life: {job_profile.get('dayInTheLife', {}).get('text', 'N/A')}\n"

        description += "- Reasons Liked:\n"
        for reason in job_profile.get('reasonsLiked', []):
            description += f"  * {reason.get('reason', '')}\n"

        description += "- Reasons Disliked:\n"
        for reason in job_profile.get('reasonsDisliked', []):
            description += f"  * {reason.get('reason', '')}\n"

        prepare_for_role = job_profile.get('prepareForRole', {})
        description += f"\nEducation Needed: {prepare_for_role.get('educationVsDegree', 'N/A')}\n"
        description += f"Training Needed: {prepare_for_role.get('trainingNeeded', 'N/A')}\n"
        description += f"Prior Work Experience: {prepare_for_role.get('priorWorkExperience', 'N/A')}\n"

    def sort_ratings(ratings):
        return sorted(ratings, key=lambda x: int(x.get('score', 0)), reverse=True)

    description += "\nAptitude Ratings:\n"
    aptitude_ratings = sort_ratings(element.get('aptitudeRatings', []))
    for rating in aptitude_ratings:
        description += f"- {rating.get('attribute', '')}, score: {rating.get('score', '')}\n"
        description += f"  Reason: {rating.get('reason', '')}\n"

    description += "\nGeographic Job Details:\n"
    for geo_detail in element.get('geographicJobDetails', []):
        description += f"- {geo_detail.get('geographicOption', '')}\n"
        description += f"  Job Availability: {geo_detail.get('jobAvailability', '')}\n"
        description += f"  Estimated Salary Range: {geo_detail.get('estimatedSalaryRange', '')}\n"

    description += "\nInterest Ratings:\n"
    interest_ratings = sort_ratings(element.get('interestRatings', []))
    for rating in interest_ratings:
        description += f"- {rating.get('attribute', '')}, score: {rating.get('score', '')}\n"
        description += f"  Reason: {rating.get('reason', '')}\n"

    description += "\nValue Ratings:\n"
    value_ratings = sort_ratings(element.get('valueRatings', []))
    for rating in value_ratings:
        description += f"- {rating.get('attribute', '')}, score: {rating.get('score', '')}\n"
        description += f"  Reason: {rating.get('reason', '')}\n"

    description += "\nCareer Pathways:\n"
    for pathway in element.get('careerPathways', []):
        description += f"- {pathway.get('pathwayTitle', '')}\n"
        for job in pathway.get('jobRoles', []):
            description += f"  * {job.get('title', '')}: {job.get('years', '')} years\n"
        description += f"  Description: {pathway.get('description', '')}\n"

    description += "\nWell-Known Employers:\n"
    for employer in element.get('employers', {}).get('wellKnownEmployers', []):
        description += f"- {employer.get('name', '')}\n"
        description += f"  Description: {employer.get('description', '')}\n"
        description += f"  Website: {employer.get('website', '')}\n"

    description += "\nEmployer Profiles:\n"
    for profile in element.get('employers', {}).get('employerProfiles', []):
        description += f"- {profile.get('geographicOption', '')}\n"
        description += f"  {profile.get('profiles', '')}\n"

    return description


def sanitize_filename(filename):
    # Remove or replace characters not suitable for filenames
    return re.sub(r'[^\w\-_\. ]', '_', filename).replace(" ", "_").lower()


def process_json_content(json_content):
    """
    Process JSON content and return a list of formatted data.

    :param json_content: JSON data containing job information (already parsed Python object)
    :return: List containing processed data
    """
    result = []
    filename_count = defaultdict(int)

    for element in json_content:
        # Get ID
        id = element.get('_id', {}).get('$oid', 'unknown_id')

        # Generate description
        text = process_json_element(element)

        # Generate filename as title
        sector = sanitize_filename(element.get('sector', 'Unknown_Sector'))
        sub_sector = sanitize_filename(element.get('subSector', 'Unknown_SubSector'))
        job_role = sanitize_filename(element.get('jobRole', 'Unknown_Role'))
        base_filename = f"{sector}_{sub_sector}_{job_role}"

        # Truncate filename if it's too long
        if len(base_filename) > 200:
            base_filename = base_filename[:200]

        # Handle duplicate filenames
        if filename_count[base_filename] > 0:
            title = f"{base_filename}_{filename_count[base_filename]}"
        else:
            title = f"{base_filename}"

        filename_count[base_filename] += 1

        # Add to result list
        result.append({
            "id": id,
            "text": text,
            "title": title
        })

    return result
